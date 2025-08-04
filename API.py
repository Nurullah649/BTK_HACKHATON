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

# --- SABİTLER VE YAPILANDIRMA ---
# Bu sabitler, kod içinde tekrar eden metinleri ve sayıları merkezi bir yerden yönetmeyi sağlar.
# Bu sayede hem okunabilirlik artar hem de gelecekteki değişiklikler kolaylaşır.
API_KEY="""AIzaSyAnI7dxlH0isxzqwqX-qkajlg2UC4zIssU"""  # UYARI: Güvenli bir yöntem değildir. Ortam değişkeni tercih edilmelidir.
GENERATION_MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "models/text-embedding-004"
DB_PATH = "urun_veritabani"
CATEGORIES_FILENAME = "kategoriler.json"

# Arama tipleri için sabitler
SEARCH_TYPE_DEFAULT = 'default'
SEARCH_TYPE_FP = 'price_performance'
SEARCH_TYPE_CHEAPEST = 'cheapest'
SEARCH_TYPE_EXPENSIVE = 'most_expensive'

# ChromaDB sorgu sonuç sayısı için sabitler
DEFAULT_N_RESULTS = 20
SPECIAL_N_RESULTS = 50

# Fiyat aralığı belirleme için sabit
PRICE_RANGE_MULTIPLIER = 0.2  # Tek fiyat bulunduğunda %20 altı ve üstü aralık oluşturur.

# --- UYGULAMA BAŞLANGICI ---
app = Flask(__name__)
CORS(app)

# --- GLOBAL DEĞİŞKENLER ---
CLIENT = None
MODEL = None
ALL_CATEGORIES = []


# --- YARDIMCI FONKSİYONLAR ---

def sanitize_collection_name(name):
    """Kategori adını veritabanı koleksiyon adı formatına çevirir."""
    # Boşlukları alt çizgi ile değiştirir ve küçük harfe çevirir.
    name = re.sub(r'\s+', '_', name)
    # Türkçe karakterleri ve geçersiz olabilecek diğer karakterleri temizler.
    name = re.sub(r'[^\w-]', '', name.lower())
    return name


def initialize_services():
    """API ve Veritabanı istemcilerini başlatır, kategorileri yükler."""
    global CLIENT, MODEL, ALL_CATEGORIES
    if not API_KEY:
        print("HATA: API_KEY bulunamadı. Lütfen API anahtarınızı tanımlayın.")
        sys.exit(1)
    try:
        print("Servisler başlatılıyor...")
        genai.configure(api_key=API_KEY)
        CLIENT = chromadb.PersistentClient(path=DB_PATH)
        MODEL = genai.GenerativeModel(GENERATION_MODEL)

        # Kategorileri dosyadan yükle
        with open(CATEGORIES_FILENAME, 'r', encoding='utf-8') as f:
            categories_list = json.load(f)
            # Doğru eşleşme için en uzun kategori adından başlayarak sırala.
            # Bu, "Ekran Kartı" gibi bir ifadenin "Kart" olarak yanlış eşleşmesini önler.
            ALL_CATEGORIES = sorted(categories_list, key=len, reverse=True)

        print("✅ Servisler başarıyla başlatıldı.")
        print(f"Veritabanı yolu: {DB_PATH}")
        print(f"{len(ALL_CATEGORIES)} kategori yüklendi.")

    except FileNotFoundError:
        print(f"HATA: '{CATEGORIES_FILENAME}' dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
        sys.exit(1)
    except Exception as e:
        print(f"HATA: Servisler başlatılamadı: {e}")
        sys.exit(1)


def parse_turkish_price(price_str):
    """Türkçe fiyat formatını (örn: "1.234,56") float sayıya çevirir."""
    if not isinstance(price_str, str):
        return float(price_str)
    try:
        # Binlik ayırıcıyı kaldırır, ondalık ayırıcıyı noktaya çevirir.
        cleaned_str = price_str.replace('.', '').replace(',', '.')
        return float(cleaned_str)
    except (ValueError, TypeError):
        return 0.0


def extract_query_details(query_text):
    """Kullanıcı sorgusunu analiz ederek kategoriyi, arama metnini ve filtreleri çıkarır."""
    print(f"Sorgu analizi başlatıldı: '{query_text}'")
    lower_query = query_text.lower()

    # 1. Kategori Tespiti
    target_collection = None
    found_category = None
    search_text = lower_query  # Varsayılan olarak tüm sorguyu arama metni yap

    for category in ALL_CATEGORIES:
        # (?<!\w) ve (?!\w) kullanarak tam kelime eşleşmesi sağlanır.
        # Bu, "kart" kelimesinin "ekran kartı" içinde eşleşmesini engeller.
        pattern = r'(?<!\w)' + re.escape(category.lower()) + r'(?!\w)'
        if re.search(pattern, lower_query):
            found_category = category
            target_collection = sanitize_collection_name(found_category)
            print(f"Kategori bulundu: '{found_category}' -> Koleksiyon: '{target_collection}'")
            # Kategori adı sorgudan temizlenerek asıl arama metni bulunur.
            search_text = re.sub(pattern, '', lower_query, count=1, flags=re.IGNORECASE).strip()
            # Eğer temizlik sonrası metin boş kalırsa, arama metni olarak kategori adının kendisi kullanılır.
            if not search_text:
                search_text = found_category.lower()
            break  # İlk ve en uzun eşleşme bulunduğunda döngüden çık

    if not target_collection:
        print(f"UYARI: Sorgu '{query_text}' içinde bilinen bir kategori bulunamadı.")

    # 2. Arama Tipi Tespiti
    search_type = SEARCH_TYPE_DEFAULT
    if 'fiyat performans' in lower_query or 'f/p' in lower_query:
        search_type = SEARCH_TYPE_FP
    elif 'en ucuz' in lower_query:
        search_type = SEARCH_TYPE_CHEAPEST
    elif 'en pahalı' in lower_query:
        search_type = SEARCH_TYPE_EXPENSIVE
    print(f"Arama tipi belirlendi: '{search_type}'")

    # 3. Fiyat Filtresi Tespiti
    where_filter = {}
    try:
        # Sorgudaki tüm sayısal değerleri bul
        prices_str = re.findall(r'(\d[\d\.,]*)', query_text)
        # Sayıları parse et ve sırala
        prices = sorted([p for p in (parse_turkish_price(s) for s in prices_str) if
                         p > 10])  # Çok küçük sayıları (model no vb.) ele

        if len(prices) >= 2:  # İki fiyat varsa aralık olarak al
            where_filter = {"$and": [{"min_price": {"$gte": prices[0]}}, {"min_price": {"$lte": prices[1]}}]}
        elif len(prices) == 1:  # Tek fiyat varsa etrafında bir aralık oluştur
            price_val = prices[0]
            lower_bound = price_val * (1 - PRICE_RANGE_MULTIPLIER)
            upper_bound = price_val * (1 + PRICE_RANGE_MULTIPLIER)
            where_filter = {"$and": [{"min_price": {"$gte": lower_bound}}, {"min_price": {"$lte": upper_bound}}]}

        if where_filter:
            print(f"Fiyat filtresi oluşturuldu: {where_filter}")
    except Exception as e:
        print(f"Fiyat filtresi oluşturulurken hata: {e}")

    return {
        "collection": target_collection,
        "where_filter": where_filter,
        "search_text": search_text.strip(),
        "search_type": search_type
    }


def get_best_product_match(client, query_details):
    """Veritabanında, arama tipine göre en uygun ürünü bulur."""
    collection_name = query_details["collection"]
    if not collection_name:
        return None, "Lütfen sorgunuzda bir ürün kategorisi belirtin."

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"HATA: '{collection_name}' koleksiyonu bulunamadı veya erişilemedi: {e}")
        return None, f"'{collection_name}' adında bir ürün kategorisi veritabanında bulunamadı."

    try:
        n_results = SPECIAL_N_RESULTS if query_details['search_type'] != SEARCH_TYPE_DEFAULT else DEFAULT_N_RESULTS

        print(f"Embedding için kullanılan arama metni: '{query_details['search_text']}'")
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=[query_details["search_text"]],
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = result['embedding']

        results = collection.query(
            query_embeddings=[query_embedding],  # Embedding bir liste içinde olmalı
            n_results=n_results,
            where=query_details["where_filter"] if query_details["where_filter"] else None
        )

        if not results or not results.get('metadatas') or not results['metadatas'][0]:
            return None, "Bu kriterlere uygun ürün bulunamadı."

        candidates = results['metadatas'][0]
        search_type = query_details['search_type']

        print(f"'{search_type}' tipine göre en iyi ürün seçiliyor...")

        if search_type == SEARCH_TYPE_FP:
            best_product = max(
                (p for p in candidates if p.get('min_price', 0) > 0),
                key=lambda p: len(p.get('features', '')) / p['min_price'],
                default=None
            )
            if not best_product: return None, "Fiyat/Performans değerlendirmesi için uygun ürün bulunamadı."
            return best_product, "Başarılı"

        elif search_type in [SEARCH_TYPE_CHEAPEST, SEARCH_TYPE_EXPENSIVE]:
            valid_price_products = [p for p in candidates if p.get('min_price', 0) > 0]
            if not valid_price_products: return None, "Fiyat bilgisi olan ürün bulunamadı."

            is_reverse = True if search_type == SEARCH_TYPE_EXPENSIVE else False
            sorted_products = sorted(valid_price_products, key=lambda p: p['min_price'], reverse=is_reverse)
            return sorted_products[0], "Başarılı"

        else:  # SEARCH_TYPE_DEFAULT
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
    except (json.JSONDecodeError, TypeError):
        offers = []

    satici_bilgisi_listesi = []
    en_ucuz_satici_linki = urun_linki  # Varsayılan link

    valid_offers = sorted(
        [o for o in offers if o.get('price') is not None and o.get('price') > 0],
        key=lambda x: x['price']
    )

    if valid_offers:
        en_ucuz_satici_linki = valid_offers[0].get('offer_url', urun_linki)
        for offer in valid_offers:
            satici_bilgisi_listesi.append(
                f"- {offer.get('seller_name', 'N/A')}: {offer.get('price', 'N/A')} TL"
            )

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
    """Gelen POST isteklerini işleyen ana API fonksiyonu."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Geçersiz istek: 'query' alanı eksik."}), 400

    user_question = data['query']
    print(f"\n--- Yeni İstek Alındı: {user_question} ---")

    # 1. Sorguyu analiz et
    query_details = extract_query_details(user_question)

    # 2. Kategori bulunamadıysa kullanıcıya öneride bulun
    if not query_details["collection"]:
        sample_categories = random.sample(ALL_CATEGORIES, min(5, len(ALL_CATEGORIES)))
        kategori_onerisi = (f"Sorgunuzda anlayabildiğim bir kategori bulamadım. "
                            f"Lütfen aşağıdaki gibi bir kategori ekleyerek tekrar deneyin:\n\n"
                            f"Örnekler: '{', '.join(sample_categories)}'")
        return jsonify({"answer": kategori_onerisi, "product_context": None})

    # 3. En uygun ürünü bul
    product_context, status = get_best_product_match(CLIENT, query_details)

    if not product_context:
        return jsonify(
            {"answer": f"Üzgünüm, bu isteğe uygun bir ürün bulamadım. (Sebep: {status})", "product_context": None})

    # 4. Nihai prompt'u oluştur ve modeli çağır
    final_prompt = generate_final_prompt(user_question, product_context)

    try:
        response = MODEL.generate_content(final_prompt)
        print("✅ Model cevabı başarıyla oluşturuldu.")
        return jsonify({"answer": response.text, "product_context": product_context})
    except Exception as e:
        print(f"HATA: Google API'den cevap alınırken bir sorun oluştu: {e}")
        return jsonify({"error": f"Google API'den cevap alınırken bir sorun oluştu: {e}"}), 500


# --- UYGULAMAYI BAŞLATMA ---
if __name__ == "__main__":
    initialize_services()
    # Gunicorn gibi bir WSGI sunucusu ile production'da çalıştırılması önerilir.
    # debug=False production için önemlidir.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
