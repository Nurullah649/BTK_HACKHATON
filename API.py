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
app = Flask(__name__)
CORS(app)

# --- KULLANICININ İSTEĞİ ÜZERİNE DEĞİŞTİRİLMEDİ ---
API_KEY = """AIzaSyAnI7dxlH0isxzqwqX-qkajlg2UC4zIssU"""
GENERATION_MODEL = "gemini-1.5-pro-latest"
# ---------------------------------------------

# --- CLOUD RUN İÇİN GEREKLİ DÜZELTME ---
DB_PATH = "urun_veritabani"
CATEGORIES_FILENAME = "kategoriler.json"
# -----------------------------------------

EMBEDDING_MODEL = "models/text-embedding-004"

# --- Global Değişkenler ---
CLIENT = None
MODEL = None
AVAILABLE_COLLECTIONS = []
ALL_CATEGORIES = []  # Kategorilerin orijinal hallerini tutan bir liste olacak

# VeriTabanı.py'den fonksiyonu import et
try:
    from VeriTabanı import sanitize_collection_name
except ImportError:
    print("HATA: VeriTabanı.py dosyası bulunamadı veya import edilemedi.")


    def sanitize_collection_name(name):
        return name.lower().replace(' ', '_')


def initialize_services():
    """API ve Veritabanı istemcilerini başlatır ve kategorileri yükler."""
    global CLIENT, MODEL, AVAILABLE_COLLECTIONS, ALL_CATEGORIES
    if not API_KEY:
        print("HATA: API_KEY bulunamadı.")
        sys.exit(1)
    try:
        print("Servisler başlatılıyor...")
        genai.configure(api_key=API_KEY)
        CLIENT = chromadb.PersistentClient(path=DB_PATH)
        MODEL = genai.GenerativeModel(GENERATION_MODEL)
        AVAILABLE_COLLECTIONS = [c.name for c in CLIENT.list_collections()]

        # Kategorileri yükle ve en uzundan kısaya doğru sırala
        with open(CATEGORIES_FILENAME, 'r', encoding='utf-8') as f:
            categories_list = json.load(f)
            # Daha doğru eşleşme için en uzun kategori adından başlayarak sırala
            ALL_CATEGORIES = sorted(categories_list, key=len, reverse=True)

        print("✅ Servisler başarıyla başlatıldı.")
        print(f"{len(ALL_CATEGORIES)} kategori yüklendi.")

    except Exception as e:
        print(f"HATA: Servisler başlatılamadı: {e}")
        sys.exit(1)


def extract_query_details(query_text, all_categories_list):
    """
    Kullanıcı sorgusunu analiz ederek kategoriyi, arama metnini ve diğer
    detayları çıkarır. Eşleştirme büyük/küçük harfe duyarsızdır.
    """
    print(f"Sorgu analizi başlatıldı: '{query_text}'")

    lower_query = query_text.lower()  # Karşılaştırma için sorguyu küçük harfe çevir
    print(f"Küçük harfe çevrilmiş sorgu: '{lower_query}'")

    target_collection = None
    found_category = None
    search_text = query_text  # Varsayılan olarak orijinal sorguyu kullan

    # Sorguda kategori adlarını ara (büyük/küçük harf duyarsız)
    # Liste zaten uzunluk sırasına göre olduğu için ilk bulunan en iyi eşleşmedir.
    for category in all_categories_list:
        if category.lower() in lower_query:
            found_category = category
            target_collection = sanitize_collection_name(found_category)
            print(f"Kategori bulundu: '{found_category}' -> Koleksiyon: '{target_collection}'")

            # Kategori adını ve olası eklerini sorgudan temizle.
            # Bu, embedding'in daha saf ürün özelliklerine odaklanmasını sağlar.
            # Örnek: "En iyi Ekran Kartları" -> "En iyi"
            cleaned_query = re.sub(r'\b' + re.escape(found_category) + r'(\w*)?\b', '', query_text,
                                   flags=re.IGNORECASE).strip()

            # Eğer temizleme sonrası sorgu anlamlı bir metin içeriyorsa onu kullan
            if len(cleaned_query.split()) > 1:  # Birden fazla kelime kaldıysa
                search_text = cleaned_query
                print(f"Sorgu temizlendi. Yeni arama metni: '{search_text}'")
            else:
                print("Temizlenmiş sorgu çok kısa, orijinal metin kullanılıyor.")

            break  # İlk ve en iyi eşleşmeyi bulduktan sonra döngüden çık

    if not target_collection:
        print(f"UYARI: Sorgu '{query_text}' içinde bilinen bir kategori bulunamadı.")

    search_type = 'default'
    if 'fiyat performans' in lower_query or 'f/p' in lower_query:
        search_type = 'price_performance'
    elif 'en ucuz' in lower_query:
        search_type = 'cheapest'
    elif 'en pahalı' in lower_query:  # Orijinal Türkçe karakterle kontrol
        search_type = 'most_expensive'

    where_filter = {}
    try:
        prices_str = re.findall(r'(\d[\d\.,]*)', query_text)
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
        "search_text": search_text,  # Temizlenmiş veya orijinal arama metni
        "search_type": search_type
    }


def get_best_product_match(client, query_details):
    """Doğru koleksiyonda, arama tipine göre en uygun ürünü bulur."""
    collection_name = query_details["collection"]
    if not collection_name:
        return None, "Kategori bulunamadı"
    try:
        collection = client.get_collection(name=collection_name)
        n_results = 50 if query_details['search_type'] != 'default' else 20

        print(f"Embedding için kullanılan arama metni: '{query_details['search_text']}'")
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
            if not valid_price_products: return None, "Fiyat bilgisi olan ürün bulunamadı"
            is_reverse = True if search_type == 'most_expensive' else False
            sorted_products = sorted(valid_price_products, key=lambda p: p['min_price'], reverse=is_reverse)
            return sorted_products[0], "Başarılı"
        else:
            return candidates[0], "Başarılı"
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


@app.route('/chat', methods=['POST'])
def chat_handler():
    """Gelen POST isteklerini işleyen ana API fonksiyonu."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Geçersiz istek: 'query' alanı eksik."}), 400
    user_question = data['query']
    print(f"\n--- Yeni İstek Alındı: {user_question} ---")

    query_details = extract_query_details(user_question, ALL_CATEGORIES)

    if not query_details["collection"]:
        sample_categories = random.sample(ALL_CATEGORIES, min(5, len(ALL_CATEGORIES))) if ALL_CATEGORIES else [
            "Örnek Kategori"]
        kategori_onerisi = f"Sorgunuzda bir kategori belirtmediniz veya anlayamadım. Lütfen sorgunuza bir kategori ekleyin. Örnekler: {', '.join(sample_categories)}"
        return jsonify({"answer": kategori_onerisi, "product_context": None})

    product_context, status = get_best_product_match(CLIENT, query_details)

    if not product_context:
        return jsonify(
            {"answer": f"Üzgünüm, bu isteğe uygun bir ürün bulamadım. (Sebep: {status})", "product_context": None})

    final_prompt = generate_final_prompt(user_question, product_context)

    try:
        response = MODEL.generate_content(final_prompt)
        print(f"Model Cevabı Oluşturuldu. Sonuç başarılı.")
        return jsonify({"answer": response.text, "product_context": product_context})
    except Exception as e:
        print(f"HATA: Google API'den cevap alınırken bir sorun oluştu: {e}")
        return jsonify({"error": f"Google API'den cevap alınırken bir sorun oluştu: {e}"}), 500


# --- UYGULAMAYI BAŞLATMA ---
if __name__ == "__main__":
    initialize_services()
    # Cloud Run'ın verdiği PORT çevre değişkenini kullanır.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
