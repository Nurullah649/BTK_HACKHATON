# =================================================================
# === UYGULAMA BAŞLANGIÇ TESTİ - BU MESAJI LOGLARDA GÖRMELİSİNİZ ===
print(">>> KOD DOSYASI BAŞARIYLA ÇALIŞTIRILDI - VERSİYON KONTROLÜ <<<")
# =================================================================

import chromadb
import google.generativeai as genai
import os
import sys
import json
import time
import re
import random
import shutil
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
# GCS kütüphanesini import et
from google.cloud import storage

# --- SABİTLER VE YAPILANDIRMA ---
API_KEY = "AIzaSyAnI7dxlH0isxzqwqX-qkajlg2UC4zIssU"  # UYARI: Güvenli bir yöntem değildir. Ortam değişkeni tercih edilmelidir.
GENERATION_MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "models/text-embedding-004"
CATEGORIES_FILENAME = "kategoriler.json"

# DÜZELTME: Bucket adı doğrudan koda eklendi.
GCS_BUCKET_NAME = "rag-api-veritabani"

# Veritabanının Cloud Run içinde indirileceği geçici konum
DB_PATH = "/tmp/urun_veritabani"
# Veritabanının Cloud Storage'daki klasör adı
DB_SOURCE_FOLDER = "urun_veritabani"

# Arama tipleri için sabitler
SEARCH_TYPE_DEFAULT = 'default'
SEARCH_TYPE_FP = 'price_performance'
SEARCH_TYPE_CHEAPEST = 'cheapest'
SEARCH_TYPE_EXPENSIVE = 'most_expensive'

# ChromaDB sorgu sonuç sayısı için sabitler
DEFAULT_N_RESULTS = 20
SPECIAL_N_RESULTS = 50

# Fiyat aralığı belirleme için sabit
PRICE_RANGE_MULTIPLIER = 0.2

# --- UYGULAMA BAŞLANGICI ---
app = Flask(__name__)
CORS(app)

# --- GLOBAL DEĞİŞKENLER ---
CLIENT = None
MODEL = None
ALL_CATEGORIES = []


# --- YARDIMCI FONKSİYONLAR ---

def download_database_from_gcs():
    """Veritabanı dosyalarını Google Cloud Storage'dan indirir."""
    bucket_name = GCS_BUCKET_NAME
    print(f"'{bucket_name}' bucket'ından veritabanı indirilmeye başlanıyor...")

    # Eğer hedef klasör varsa, temizle
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    os.makedirs(DB_PATH)

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=DB_SOURCE_FOLDER)  # Sadece ilgili klasördeki dosyaları listele

        file_count = 0
        for blob in blobs:
            if not blob.name.endswith('/'):  # Klasörleri atla
                destination_file_name = os.path.join(DB_PATH, os.path.basename(blob.name))
                blob.download_to_filename(destination_file_name)
                file_count += 1

        if file_count == 0:
            print(f"UYARI: '{bucket_name}/{DB_SOURCE_FOLDER}' içinde indirilecek dosya bulunamadı.")
            return False

        print(f"✅ Veritabanı başarıyla indirildi. Toplam {file_count} dosya indirildi -> {DB_PATH}")
        return True
    except Exception as e:
        print(f"HATA: GCS'den veritabanı indirilirken bir sorun oluştu: {e}")
        return False


def sanitize_collection_name(name):
    """
    Kategori adını, veritabanını oluşturan kod ile %100 uyumlu olacak şekilde
    ChromaDB için geçerli bir koleksiyon adına dönüştürür.
    """
    if not name:
        return "diger_kategoriler"
    char_map = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
    }
    for tr_char, en_char in char_map.items():
        name = name.replace(tr_char, en_char)
    name = name.lower()
    name = name.replace(' ', '_').replace('-', '_')
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = name.strip('_')
    if len(name) < 3: name = f"{name}_koleksiyonu"
    if len(name) > 63: name = name[:63]
    return name if name else "diger_kategoriler"


def initialize_services():
    """API ve Veritabanı istemcilerini başlatır, kategorileri yükler."""
    global CLIENT, MODEL, ALL_CATEGORIES
    print(">>> initialize_services fonksiyonu çağrıldı.")
    if not API_KEY:
        print("HATA: API_KEY bulunamadı.")
        sys.exit(1)

    try:
        # 1. ADIM: Veritabanını GCS'den indir
        if not download_database_from_gcs():
            sys.exit("Veritabanı indirilemediği için uygulama durduruluyor.")

        # 2. ADIM: Servisleri başlat
        print("Servisler başlatılıyor...")
        genai.configure(api_key=API_KEY)
        CLIENT = chromadb.PersistentClient(path=DB_PATH)  # İndirilen yolu kullan
        MODEL = genai.GenerativeModel(GENERATION_MODEL)

        # 3. ADIM: Kategorileri yükle
        with open(CATEGORIES_FILENAME, 'r', encoding='utf-8') as f:
            categories_list = json.load(f)
        ALL_CATEGORIES = sorted(categories_list, key=len, reverse=True)

        print("✅ Servisler başarıyla başlatıldı.")
        print(f"Veritabanı konumu: {DB_PATH}")
        print(f"{len(ALL_CATEGORIES)} kategori yüklendi.")

    except FileNotFoundError:
        print(f"HATA: '{CATEGORIES_FILENAME}' dosyası bulunamadı.")
        sys.exit(1)
    except Exception as e:
        print(f"HATA: Servisler başlatılamadı: {e}")
        sys.exit(1)


def parse_turkish_price(price_str):
    if not isinstance(price_str, str): return float(price_str)
    try:
        return float(price_str.replace('.', '').replace(',', '.'))
    except (ValueError, TypeError):
        return 0.0


def convert_words_to_numbers(text):
    text = re.sub(r'(\d[\d\.,]*)\s*(?:bin|k)\b', lambda m: str(int(parse_turkish_price(m.group(1)) * 1000)), text,
                  flags=re.IGNORECASE)
    text = re.sub(r'(\d[\d\.,]*)\s*milyon\b', lambda m: str(int(parse_turkish_price(m.group(1)) * 1000000)), text,
                  flags=re.IGNORECASE)
    text = re.sub(r'\byüz\s+bin\b', '100000', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbir\s+milyon\b', '1000000', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbin\b', '1000', text, flags=re.IGNORECASE)
    return text


def extract_query_details(query_text):
    print(f"Orijinal sorgu: '{query_text}'")
    processed_query = convert_words_to_numbers(query_text)
    print(f"Fiyat dönüşümü sonrası sorgu: '{processed_query}'")
    lower_query = processed_query.lower()
    target_collection = None
    found_category = None
    search_text = processed_query
    for category in ALL_CATEGORIES:
        category_lower = category.strip().lower()
        if category_lower in lower_query:
            found_category = category
            target_collection = sanitize_collection_name(found_category)
            print(f"Kategori bulundu: '{found_category}' -> Koleksiyon: '{target_collection}'")
            break
    if not target_collection:
        print(f"UYARI: Sorgu '{processed_query}' içinde bilinen bir kategori bulunamadı.")
    search_type = SEARCH_TYPE_DEFAULT
    if 'fiyat performans' in lower_query or 'f/p' in lower_query:
        search_type = SEARCH_TYPE_FP
    elif 'en ucuz' in lower_query:
        search_type = SEARCH_TYPE_CHEAPEST
    elif 'en pahalı' in lower_query:
        search_type = SEARCH_TYPE_EXPENSIVE
    print(f"Arama tipi belirlendi: '{search_type}'")
    where_filter = {}
    try:
        prices_str = re.findall(r'(\d[\d\.,]*)', processed_query)
        prices = sorted([p for p in (parse_turkish_price(s) for s in prices_str) if p > 10])
        if len(prices) >= 2:
            where_filter = {"$and": [{"min_price": {"$gte": prices[0]}}, {"min_price": {"$lte": prices[1]}}]}
        elif len(prices) == 1:
            price_val = prices[0]
            lower_bound = price_val * (1 - PRICE_RANGE_MULTIPLIER)
            upper_bound = price_val * (1 + PRICE_RANGE_MULTIPLIER)
            where_filter = {"$and": [{"min_price": {"$gte": lower_bound}}, {"min_price": {"$lte": upper_bound}}]}
        if where_filter: print(f"Fiyat filtresi oluşturuldu: {where_filter}")
    except Exception as e:
        print(f"Fiyat filtresi oluşturulurken hata: {e}")
    return {"collection": target_collection, "where_filter": where_filter, "search_text": search_text.strip(),
            "search_type": search_type}


def get_best_product_match(client, query_details):
    collection_name = query_details["collection"]
    if not collection_name: return None, "Lütfen sorgunuzda bir ürün kategorisi belirtin."
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"HATA: '{collection_name}' koleksiyonu bulunamadı veya erişilemedi: {e}")
        return None, f"'{collection_name}' adında bir ürün kategorisi veritabanında bulunamadı."
    try:
        n_results = SPECIAL_N_RESULTS if query_details['search_type'] != SEARCH_TYPE_DEFAULT else DEFAULT_N_RESULTS
        print(f"Embedding için kullanılan arama metni: '{query_details['search_text']}'")
        result = genai.embed_content(model=EMBEDDING_MODEL, content=[query_details["search_text"]],
                                     task_type="RETRIEVAL_QUERY")
        query_embedding = result['embedding']

        # DÜZELTME: ChromaDB'ye gönderilen embedding formatı düzeltildi.
        # Fazladan liste parantezleri kaldırıldı.
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=query_details["where_filter"] if query_details["where_filter"] else None
        )

        if not results or not results.get('metadatas') or not results['metadatas'][0]:
            return None, "Bu kriterlere uygun ürün bulunamadı."
        candidates = results['metadatas'][0]
        search_type = query_details['search_type']
        print(f"'{search_type}' tipine göre en iyi ürün seçiliyor...")
        if search_type == SEARCH_TYPE_FP:
            best_product = max((p for p in candidates if p.get('min_price', 0) > 0),
                               key=lambda p: len(p.get('features', '')) / p['min_price'], default=None)
            if not best_product: return None, "Fiyat/Performans değerlendirmesi için uygun ürün bulunamadı."
            return best_product, "Başarılı"
        elif search_type in [SEARCH_TYPE_CHEAPEST, SEARCH_TYPE_EXPENSIVE]:
            valid_price_products = [p for p in candidates if p.get('min_price', 0) > 0]
            if not valid_price_products: return None, "Fiyat bilgisi olan ürün bulunamadı."
            is_reverse = True if search_type == SEARCH_TYPE_EXPENSIVE else False
            sorted_products = sorted(valid_price_products, key=lambda p: p['min_price'], reverse=is_reverse)
            return sorted_products[0], "Başarılı"
        else:
            return candidates[0], "Başarılı"
    except Exception as e:
        print(f"HATA: Ürün arama sırasında bir sorun oluştu: {e}")
        return None, "Arama sırasında beklenmedik bir sorun oluştu."


def generate_final_prompt(user_question, product_context):
    urun_adi = product_context.get('product_name', 'N/A')
    urun_linki = product_context.get('product_url', 'N/A')
    urun_ozellikleri = product_context.get('features', 'Mevcut değil.')
    try:
        offers = json.loads(product_context.get('offers_json', '[]'))
    except (json.JSONDecodeError, TypeError):
        offers = []
    satici_bilgisi_listesi = []
    en_ucuz_satici_linki = urun_linki
    valid_offers = sorted([o for o in offers if o.get('price') is not None and o.get('price') > 0],
                          key=lambda x: x['price'])
    if valid_offers:
        en_ucuz_satici_linki = valid_offers[0].get('offer_url', urun_linki)
        for offer in valid_offers:
            satici_bilgisi_listesi.append(f"- {offer.get('seller_name', 'N/A')}: {offer.get('price', 'N/A')} TL")
    satici_bilgisi = "\n".join(
        satici_bilgisi_listesi) if satici_bilgisi_listesi else "Online satıcı bilgisi bulunamadı."
    prompt = f"""
Sen, müşteri memnuniyetini ve satışı en üst düzeye çıkarmayı hedefleyen, son derece bilgili, ikna edici ve yardımcı bir "Akıllı Satış Asistanı"sın.
### SAĞLANAN BİLGİLER (RAG) ###
- **Ürün Adı:** {urun_adi}
- **Ürün Özellikleri:** {urun_ozellikleri}
- **Fiyat Bilgileri:**
{satici_bilgisi}
- **En Uygun Fiyatlı Satıcı Linki:** {en_ucuz_satici_linki}
### DAVRANIŞ KURALLARI ###
1.  **Tek Bilgi Kaynağı:** Cevaplarını oluştururken SADECE ### SAĞLANAN BİLGİLER (RAG) ### bölümündeki verileri kullan. DIŞARIDAN BİLGİ EKLEME.
2.  **Bilgi Eksikliği:** Eğer kullanıcının sorusunun cevabı sağlanan bilgilerde yoksa, "Bu konuda elimdeki bilgilerde net bir cevap bulamadım." gibi bir ifade kullan. Asla bilgi uydurma.
3.  **Özellikten Faydaya:** Ürün özelliklerini, müşterinin hayatına katacağı değere ve faydaya dönüştürerek, basit ve anlaşılır bir dille açıkla. Teknik jargonlardan kaçın.
4.  **Eylem Çağrısı:** Cevabının sonunda, müşteriyi ürünü incelemeye veya satın almaya teşvik etmek için net bir yönlendirme yap. Örneğin: "Ürünü daha detaylı incelemek ve en uygun fiyatla satın almak için bu linki kullanabilirsiniz: {en_ucuz_satici_linki}"
### KULLANICI SORUSU ###
{user_question}
"""
    return prompt


# --- API ENDPOINT ---
@app.route('/chat', methods=['POST'])
def chat_handler():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Geçersiz istek: 'query' alanı eksik."}), 400
    user_question = data['query']
    print(f"\n--- Yeni İstek Alındı: {user_question} ---")
    query_details = extract_query_details(user_question)
    if not query_details["collection"]:
        sample_categories = random.sample(ALL_CATEGORIES, min(5, len(ALL_CATEGORIES)))
        kategori_onerisi = (f"Sorgunuzda anlayabildiğim bir kategori bulamadım. "
                            f"Lütfen aşağıdaki gibi bir kategori ekleyerek tekrar deneyin:\n\n"
                            f"Örnekler: '{', '.join(sample_categories)}'")
        return jsonify({"answer": kategori_onerisi, "product_context": None})
    product_context, status = get_best_product_match(CLIENT, query_details)
    if not product_context:
        return jsonify(
            {"answer": f"Üzgünüm, bu isteğe uygun bir ürün bulamadım. (Sebep: {status})", "product_context": None})
    final_prompt = generate_final_prompt(user_question, product_context)
    try:
        response = MODEL.generate_content(final_prompt)
        print("✅ Model cevabı başarıyla oluşturuldu.")
        return jsonify({"answer": response.text, "product_context": product_context})
    except Exception as e:
        print(f"HATA: Google API'den cevap alınırken bir sorun oluştu: {e}")
        return jsonify({"error": f"Google API'den cevap alınırken bir sorun oluştu: {e}"}), 500


# --- UYGULAMAYI BAŞLATMA ---
initialize_services()

if __name__ == "__main__":
    print(">>> __main__ bloğu çalıştırılıyor (YEREL GELİŞTİRME).")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
