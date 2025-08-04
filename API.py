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
# UYARI: API anahtarını koda bu şekilde yazmak GÜVENLİ DEĞİLDİR.
# Bu kodun bir sonraki versiyonunda bu anahtarın ortam değişkenlerinden (environment variables)
# yüklenmesi şiddetle tavsiye edilir.
API_KEY="""AIzaSyAnI7dxlH0isxzqwqX-qkajlg2UC4zIssU"""
GENERATION_MODEL = "gemini-2.5-pro"
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
ALL_CATEGORIES = []

# VeriTabanı.py'den fonksiyonu import et
try:
    from VeriTabanı import sanitize_collection_name
except ImportError:
    print("UYARI: VeriTabanı.py dosyası bulunamadı. Standart fonksiyon kullanılacak.")
    def sanitize_collection_name(name):
        return re.sub(r'\s+', '_', name.lower())


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

        with open(CATEGORIES_FILENAME, 'r', encoding='utf-8') as f:
            categories_list = json.load(f)
            # Daha doğru eşleşme için en uzun kategori adından başlayarak sırala
            ALL_CATEGORIES = sorted(categories_list, key=len, reverse=True)

        print("✅ Servisler başarıyla başlatıldı.")
        print(f"{len(ALL_CATEGORIES)} kategori yüklendi.")

    except Exception as e:
        print(f"HATA: Servisler başlatılamadı: {e}")
        sys.exit(1)

# --- YENİ EKLENEN YARDIMCI FONKSİYONLAR ---

def convert_words_to_numbers(text):
    """Sorgu metnindeki 'bin', 'milyon' gibi sayısal ifadeleri rakamlara çevirir."""
    # Not: Bu fonksiyon çağrılmadan önce metin küçük harfe çevrilmemelidir.
    # Regex işlemleri burada case-insensitive olarak yapılabilir.
    text = re.sub(r'(\d[\d\.,]*)\s*(?:bin|k)\b', lambda m: str(int(parse_turkish_price(m.group(1)) * 1000)), text, flags=re.IGNORECASE)
    text = re.sub(r'(\d[\d\.,]*)\s*milyon\b', lambda m: str(int(parse_turkish_price(m.group(1)) * 1000000)), text, flags=re.IGNORECASE)
    text = re.sub(r'\byüz\s+bin\b', '100000', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbir\s+milyon\b', '1000000', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbin\b', '1000', text, flags=re.IGNORECASE)
    return text

def parse_turkish_price(price_str):
    """Türkçe fiyat formatını (örn: "1.234,56") float sayıya çevirir."""
    if not isinstance(price_str, str):
        return float(price_str)
    try:
        # Binlik ayırıcı olan noktaları kaldır
        cleaned_str = price_str.replace('.', '')
        # Ondalık ayırıcı olan virgülü noktaya çevir
        cleaned_str = cleaned_str.replace(',', '.')
        return float(cleaned_str)
    except (ValueError, TypeError):
        return 0.0 # Hata durumunda 0 döndür

# --- GÜNCELLENEN FONKSİYONLAR ---

def extract_query_details(query_text, all_categories_list):
    """
    Kullanıcı sorgusunu analiz ederek kategoriyi, arama metnini, fiyat aralığını
    ve diğer detayları çıkarır.
    """
    print(f"Sorgu analizi başlatıldı: '{query_text}'")

    # 1. Adım: Sorgudaki "bin", "milyon" gibi ifadeleri sayılara çevir
    processed_query = convert_words_to_numbers(query_text)
    print(f"Sayısal ifadeler işlendi: '{processed_query}'")

    target_collection = None
    found_category = None
    search_text = processed_query # Varsayılan olarak işlenmiş sorguyu kullan

    # 2. Adım: Kategori tespiti (En uzundan kısaya doğru, case-insensitive)
    for category in all_categories_list:
        # \b ile kelime sınırı kontrolü yapılır.
        # re.IGNORECASE kullanarak büyük/küçük harf duyarsız ve Türkçe karakterlere
        # duyarlı bir arama yapılır. Bu, .lower() kullanmaktan daha güvenilirdir.
        pattern = r'\b' + re.escape(category) + r'\b'
        if re.search(pattern, processed_query, re.IGNORECASE):
            found_category = category
            target_collection = sanitize_collection_name(found_category)
            print(f"Kategori bulundu: '{found_category}' -> Koleksiyon: '{target_collection}'")

            # Kategori adını sorgudan temizle
            cleaned_query = re.sub(pattern, '', processed_query, flags=re.IGNORECASE).strip()
            # Oluşabilecek çoklu boşlukları tek boşluğa indirge
            cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

            # Eğer temizleme sonrası sorguda anlamlı bir metin kaldıysa onu kullan.
            # Yoksa (örn: kullanıcı sadece "Aspiratör" dediyse), arama metni olarak
            # kategori adının kendisini kullan. Bu, embedding'in boş metinle hata vermesini önler.
            if cleaned_query:
                search_text = cleaned_query
                print(f"Sorgu temizlendi. Yeni arama metni: '{search_text}'")
            else:
                search_text = found_category
                print(f"Temizlenmiş sorgu boş. Arama metni olarak kategori adı kullanılıyor: '{search_text}'")
            break

    if not target_collection:
        print(f"UYARI: Sorgu '{processed_query}' içinde bilinen bir kategori bulunamadı.")

    # 3. Adım: Arama tipini belirle
    search_type = 'default'
    lower_processed_query = processed_query.lower() # Arama tipi için küçük harf kullan
    if 'fiyat performans' in lower_processed_query or 'f/p' in lower_processed_query:
        search_type = 'price_performance'
    elif 'en ucuz' in lower_processed_query:
        search_type = 'cheapest'
    elif 'en pahalı' in lower_processed_query:
        search_type = 'most_expensive'

    # 4. Adım: Fiyat aralığını ve filtreleri çıkar (Geliştirilmiş parser ile)
    where_filter = {}
    try:
        # Sadece sayısal karakterleri ve ayraçları içeren kısımları bul
        prices_str = re.findall(r'(\d[\d\.,]*)', processed_query)
        prices = sorted([p for p in (parse_turkish_price(s) for s in prices_str) if p > 0])
        if len(prices) >= 2:
            where_filter = {"$and": [{"min_price": {"$gte": prices[0]}}, {"min_price": {"$lte": prices[1]}}]}
        elif len(prices) == 1:
            price_val = prices[0]
            # Fiyat etrafında %20'lik bir aralık belirle
            where_filter = {"$and": [{"min_price": {"$gte": price_val * 0.8}}, {"min_price": {"$lte": price_val * 1.2}}]}
        if where_filter:
             print(f"Fiyat filtresi oluşturuldu: {where_filter}")
    except Exception as e:
        print(f"Fiyat filtresi oluşturulurken hata: {e}")


    return {
        "collection": target_collection,
        "where_filter": where_filter,
        "search_text": search_text,
        "search_type": search_type
    }


def get_best_product_match(client, query_details):
    """Doğru koleksiyonda, arama tipine göre en uygun ürünü bulur."""
    collection_name = query_details["collection"]
    if not collection_name:
        return None, "Lütfen sorgunuzda bir ürün kategorisi belirtin."
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
            return None, "Bu kriterlere uygun ürün bulunamadı."

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
        print(f"HATA: Ürün arama sırasında bir sorun oluştu: {e}")
        return None, "Arama sırasında beklenmedik bir sorun oluştu."


def generate_final_prompt(user_question, product_context):
    """Prompt şablonunu, bulunan ürün bilgileriyle doldurur."""
    urun_adi = product_context.get('product_name', 'N/A')
    urun_linki = product_context.get('product_url', 'N/A')
    urun_ozellikleri = product_context.get('features', 'Mevcut değil.')
    try:
        offers = json.loads(product_context.get('offers_json', '[]'))
    except json.JSONDecodeError:
        offers = []

    satici_bilgisi_listesi = []
    satici_linki = urun_linki
    if offers:
        valid_offers = [o for o in offers if o.get('price') is not None and o.get('price') > 0]
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
        sample_categories = random.sample(ALL_CATEGORIES, min(5, len(ALL_CATEGORIES))) if ALL_CATEGORIES else ["Örnek Kategori"]
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
