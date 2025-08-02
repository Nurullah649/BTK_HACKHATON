import chromadb
import google.generativeai as genai
import os
import sys
import json
import time
import re
import random
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- AYARLAR VE BAŞLANGIÇ ---


# Flask uygulamasını oluştur ve CORS'u etkinleştir (farklı domainlerden gelen isteklere izin verir)
app = Flask(__name__)
CORS(app)

API_KEY="""AIzaSyAnI7dxlH0isxzqwqX-qkajlg2UC4zIssU"""

DB_PATH = "../urun_veritabani"
CATEGORIES_FILENAME = "../kategoriler.json"
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-pro"

# --- Servisleri ve Verileri Global Olarak Yükleme ---
# Bu değişkenler, API çalıştığı sürece hafızada kalacak ve her istekte tekrar yüklenmeyecek.
CLIENT = None
MODEL = None
AVAILABLE_COLLECTIONS = []
ALL_CATEGORIES = []


def initialize_services():
    """API ve Veritabanı istemcilerini başlatır, mevcut koleksiyonları ve kategorileri listeler."""
    global CLIENT, MODEL, AVAILABLE_COLLECTIONS, ALL_CATEGORIES

    if not API_KEY:
        print("HATA: GOOGLE_API_KEY ortam değişkeni bulunamadı.")
        sys.exit()

    try:
        genai.configure(api_key=API_KEY)
        CLIENT = chromadb.PersistentClient(path=DB_PATH)
        MODEL = genai.GenerativeModel(GENERATION_MODEL)

        AVAILABLE_COLLECTIONS = [c.name for c in CLIENT.list_collections()]
        if not AVAILABLE_COLLECTIONS:
            print("HATA: Veritabanında hiç koleksiyon bulunamadı. Lütfen önce VeriTabanı.py betiğini çalıştırın.")
            sys.exit()

        with open(CATEGORIES_FILENAME, 'r', encoding='utf-8') as f:
            ALL_CATEGORIES = json.load(f)

        print(
            f"✅ {len(AVAILABLE_COLLECTIONS)} koleksiyon ve {len(ALL_CATEGORIES)} kategori bulundu. API servise hazır.")
    except FileNotFoundError:
        print(
            f"HATA: Kategori dosyası '{CATEGORIES_FILENAME}' bulunamadı. Lütfen önce kategori çıkarma betiğini çalıştırın.")
        sys.exit()
    except Exception as e:
        print(f"HATA: Veritabanı veya API başlatılamadı: {e}")
        sys.exit()


def extract_query_details(query_text, all_categories):
    """
    Kullanıcı sorgusunu analiz ederek kategori, arama tipi, fiyat filtresi ve arama metnini çıkarır.
    """
    query_lower = query_text.lower()

    target_collection = None
    from VeriTabanı import sanitize_collection_name
    for category in all_categories:
        if category.lower() in query_lower:
            target_collection = sanitize_collection_name(category)
            break

    search_type = 'default'
    if 'fiyat performans' in query_lower or 'f/p' in query_lower:
        search_type = 'price_performance'
    elif 'en ucuz' in query_lower:
        search_type = 'cheapest'
    elif 'en pahalı' in query_lower:
        search_type = 'most_expensive'

    where_filter = {}
    try:
        prices_str = re.findall(r'(\d[\d\.,]*)', query_lower)
        prices = sorted([float(p.replace('.', '').replace(',', '')) for p in prices_str])

        if len(prices) >= 2:
            where_filter = {"$and": [{"min_price": {"$gte": prices[0]}}, {"min_price": {"$lte": prices[1]}}]}
        elif len(prices) == 1:
            price_val = prices[0]
            price_min = price_val * 0.8
            price_max = price_val * 1.2
            where_filter = {"$and": [{"min_price": {"$gte": price_min}}, {"min_price": {"$lte": price_max}}]}
    except (ValueError, TypeError):
        pass

    return {
        "collection": target_collection,
        "where_filter": where_filter,
        "search_text": query_text,
        "search_type": search_type
    }


def get_best_product_match(client, query_details):
    """
    Doğru koleksiyonda, arama tipine göre en uygun ürünü bulur.
    """
    collection_name = query_details["collection"]
    if not collection_name:
        return None, "Kategori bulunamadı"

    try:
        collection = client.get_collection(name=collection_name)

        n_results = 50 if query_details['search_type'] != 'default' else 20

        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=[query_details["search_text"]],
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = result['embedding']

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=query_details["where_filter"] if query_details["where_filter"] else None
        )

        if not results or not results.get('metadatas') or not results['metadatas'][0]:
            return None, "Bu kriterlere uygun ürün bulunamadı"

        candidates = results['metadatas'][0]
        search_type = query_details['search_type']

        if search_type == 'price_performance':
            best_product = None
            max_score = -1
            for product in candidates:
                price = product.get('min_price', 0)
                feature_length = len(product.get('features', ''))
                if price > 0:
                    score = feature_length / price
                    if score > max_score:
                        max_score = score
                        best_product = product
            return best_product, "Başarılı"

        elif search_type in ['cheapest', 'most_expensive']:
            valid_price_products = [p for p in candidates if p.get('min_price', -1) > 0]
            if not valid_price_products:
                return None, "Fiyat bilgisi olan ürün bulunamadı"

            is_reverse = True if search_type == 'most_expensive' else False
            sorted_products = sorted(valid_price_products, key=lambda p: p['min_price'], reverse=is_reverse)
            return sorted_products[0], "Başarılı"

        else:
            return candidates[0], "Başarılı"

    except ValueError:
        return None, f"'{collection_name}' adında bir kategori bulunamadı."
    except Exception as e:
        return None, f"Arama hatası: {e}"


def generate_final_prompt(user_question, product_context):
    """Prompt şablonunu, bulunan ürün bilgileriyle doldurur."""
    urun_adi = product_context.get('product_name', 'N/A')
    urun_linki = product_context.get('product_url', 'N/A')
    urun_ozellikleri = product_context.get('features', 'Mevcut değil.')

    offers = json.loads(product_context.get('offers_json', '[]'))
    satici_bilgisi_listesi = []
    satici_linki = urun_linki

    if offers:
        valid_offers = [o for o in offers if o.get('price', 0) > 0]
        if valid_offers:
            satici_linki = min(valid_offers, key=lambda x: x['price'])['offer_url']
        for offer in sorted(valid_offers, key=lambda x: x.get('price', float('inf'))):
            satici_bilgisi_listesi.append(
                f"- {offer.get('seller_name', 'N/A')}: {offer.get('price', 'N/A')} TL ({offer.get('stock_status', 'N/A')})")

    satici_bilgisi = "\n".join(satici_bilgisi_listesi) if satici_bilgisi_listesi else "Satıcı bilgisi bulunamadı."

    prompt = f"""
Sen, müşteri memnuniyetini ve satışı en üst düzeye çıkarmayı hedefleyen, son derece bilgili, ikna edici ve yardımcı bir "Akıllı Satış Asistanı"sın.

### SAĞLANAN BİLGİLER (RAG) ###
- **Ürün Adı:** {urun_adi}
- **Ürün Özellikleri:** {urun_ozellikleri}
- **Satıcı Bilgisi:**
{satici_bilgisi}
- **En Ucuz Satıcı Linki:** {satici_linki}

### DAVRANIŞ KURALLARI ###
1.  **Öncelikli Bilgi Kaynağı:** Cevaplarını oluştururken sadece ### SAĞLANAN BİLGİLER (RAG) ### bölümündeki verileri kullan.
2.  **Bilgi Eksikliği Durumu:** Eğer kullanıcının sorusunun cevabı sağlanan bilgilerde yer almıyorsa, nazikçe, "Bu konuda elimdeki bilgilerde net bir cevap bulamadım." şeklinde yanıt ver. Asla bilgi uydurma veya internetten araştırma yapma.
3.  **Özellikten Faydaya Çeviri:** Ürün özelliklerini, müşterinin hayatına katacağı değere ve faydaya dönüştürerek açıkla.
4.  **Eylem Çağrısı:** Müşteri ürünü satın almaya veya incelemeye hazır görünüyorsa, "Ürünü daha detaylı incelemek veya güvenilir satıcıdan satın almak için bu linki kullanabilirsiniz: {satici_linki}" diyerek net bir yönlendirme yap.

### KULLANICI SORUSU ###
{user_question}
"""
    return prompt


# --- API UÇ NOKTASI (ENDPOINT) ---

@app.route('/chat', methods=['POST'])
def chat_handler():
    """
    Gelen POST isteklerini işleyen ana API fonksiyonu.
    """
    # İstekten JSON verisini al
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Geçersiz istek: 'query' alanı eksik."}), 400

    user_question = data['query']

    # Sorguyu analiz et
    query_details = extract_query_details(user_question, ALL_CATEGORIES)

    if not query_details["collection"]:
        sample_categories = random.sample(ALL_CATEGORIES, min(5, len(ALL_CATEGORIES)))
        kategori_onerisi = f"Sorgunuzda bir kategori belirtmediniz veya anlayamadım. Lütfen sorgunuza bir kategori ekleyin. Örnekler: {', '.join(repr(c) for c in sample_categories)}"
        return jsonify({"answer": kategori_onerisi, "product_context": None})

    # Ürün ara
    product_context, status = get_best_product_match(CLIENT, query_details)

    if not product_context:
        return jsonify(
            {"answer": f"Üzgünüm, bu isteğe uygun bir ürün bulamadım. (Sebep: {status})", "product_context": None})

    # Nihai prompt'u oluştur ve LLM'e gönder
    final_prompt = generate_final_prompt(user_question, product_context)

    try:
        response = MODEL.generate_content(final_prompt)
        # Cevabı ve bulunan ürünün bilgilerini JSON olarak döndür
        return jsonify({"answer": response.text, "product_context": product_context})
    except Exception as e:
        return jsonify({"error": f"Google API'den cevap alınırken bir sorun oluştu: {e}"}), 500


# --- UYGULAMAYI BAŞLATMA ---

if __name__ == "__main__":
    # Servisleri (veritabanı, model vb.) API başlamadan önce bir kez yükle
    initialize_services()
    # Flask uygulamasını başlat (debug=False production için önerilir)
    app.run(host='0.0.0.0', port=5000, debug=True)
