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
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
# GCS ve Firestore kütüphanelerini import et
from google.cloud import storage
from google.cloud import firestore

# --- SABİTLER VE YAPILANDIRMA ---
# Güvenlik için API anahtarını ortam değişkenlerinden almak en iyisidir.
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAnI7dxlH0isxzqwqX-qkajlg2UC4zIssU")
GENERATION_MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "models/text-embedding-004"
CATEGORIES_FILENAME = "kategoriler.json"
GCS_BUCKET_NAME = "rag-api-veritabani"
DB_PATH = "/tmp/urun_veritabani"
DB_SOURCE_FOLDER = "urun_veritabani"
SEARCH_TYPE_DEFAULT = 'default'
SEARCH_TYPE_FP = 'price_performance'
SEARCH_TYPE_CHEAPEST = 'cheapest'
SEARCH_TYPE_EXPENSIVE = 'most_expensive'
DEFAULT_N_RESULTS = 20
SPECIAL_N_RESULTS = 50
PRICE_RANGE_MULTIPLIER = 0.2

# --- UYGULAMA BAŞLANGICI ---
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

# --- GLOBAL DEĞİŞKENLER ---
CLIENT = None
MODEL = None
ALL_CATEGORIES = []
DB_FIRESTORE = None


# --- YARDIMCI FONKSİYONLAR ---

def download_database_from_gcs():
    """Veritabanı dosyalarını Google Cloud Storage'dan indirir."""
    bucket_name = GCS_BUCKET_NAME
    print(f"'{bucket_name}' bucket'ından veritabanı indirilmeye başlanıyor...")
    if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
    os.makedirs(DB_PATH)
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=DB_SOURCE_FOLDER)
        file_count = 0
        for blob in blobs:
            if not blob.name.endswith('/'):
                destination_file_name = os.path.join(DB_PATH, os.path.basename(blob.name))
                blob.download_to_filename(destination_file_name)
                file_count += 1
        if file_count == 0: return False
        print(f"✅ Veritabanı başarıyla indirildi. Toplam {file_count} dosya indirildi -> {DB_PATH}")
        return True
    except Exception as e:
        print(f"HATA: GCS'den veritabanı indirilirken bir sorun oluştu: {e}")
        return False


def sanitize_collection_name(name):
    if not name: return "diger_kategoriler"
    char_map = {'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c', 'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's',
                'Ö': 'o', 'Ç': 'c'}
    for tr, en in char_map.items(): name = name.replace(tr, en)
    name = name.lower()
    name = name.replace(' ', '_').replace('-', '_')
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = name.strip('_')
    if len(name) < 3: name = f"{name}_koleksiyonu"
    if len(name) > 63: name = name[:63]
    return name if name else "diger_kategoriler"


def initialize_services():
    """API ve Veritabanı istemcilerini başlatır."""
    global CLIENT, MODEL, ALL_CATEGORIES, DB_FIRESTORE
    print(">>> Servisler başlatılıyor... <<<")
    if not API_KEY: print("HATA: API_KEY bulunamadı."); sys.exit(1)
    try:
        if not download_database_from_gcs():
            sys.exit("Veritabanı indirilemediği için uygulama durduruluyor.")
        genai.configure(api_key=API_KEY)
        CLIENT = chromadb.PersistentClient(path=DB_PATH)
        MODEL = genai.GenerativeModel(GENERATION_MODEL)
        DB_FIRESTORE = firestore.Client()
        with open(CATEGORIES_FILENAME, 'r', encoding='utf-8') as f:
            categories_list = json.load(f)
        ALL_CATEGORIES = sorted(categories_list, key=len, reverse=True)
        print("✅ Servisler başarıyla başlatıldı.")
    except Exception as e:
        print(f"HATA: Servisler başlatılamadı: {e}");
        sys.exit(1)


def get_conversation_history(session_id):
    try:
        doc = DB_FIRESTORE.collection('conversations').document(session_id).get()
        return doc.to_dict().get('history', []) if doc.exists else []
    except Exception as e:
        print(f"HATA: Konuşma geçmişi alınamadı: {e}");
        return []


def save_conversation_history(session_id, history):
    try:
        DB_FIRESTORE.collection('conversations').document(session_id).set({'history': history})
    except Exception as e:
        print(f"HATA: Konuşma geçmişi kaydedilemedi: {e}")


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
    return text


def extract_query_details(query_text):
    processed_query = convert_words_to_numbers(query_text)
    lower_query = processed_query.lower()
    target_collection, found_category, search_text = None, None, processed_query
    for category in ALL_CATEGORIES:
        category_lower = category.strip().lower()
        if category_lower in lower_query:
            found_category, target_collection = category, sanitize_collection_name(category)
            break
    search_type = SEARCH_TYPE_DEFAULT
    if 'fiyat performans' in lower_query or 'f/p' in lower_query:
        search_type = SEARCH_TYPE_FP
    elif 'daha ucuz' in lower_query or 'en ucuz' in lower_query:
        search_type = SEARCH_TYPE_CHEAPEST
    elif 'daha pahalı' in lower_query or 'en pahalı' in lower_query:
        search_type = SEARCH_TYPE_EXPENSIVE
    where_filter = {}
    try:
        prices_str = re.findall(r'(\d[\d\.,]*)', processed_query)
        prices = sorted([p for p in (parse_turkish_price(s) for s in prices_str) if p > 10])
        if len(prices) >= 2:
            where_filter = {"$and": [{"min_price": {"$gte": prices[0]}}, {"min_price": {"$lte": prices[1]}}]}
        elif len(prices) == 1:
            price_val = prices[0]
            where_filter = {"$and": [{"min_price": {"$gte": price_val * (1 - PRICE_RANGE_MULTIPLIER)}},
                                     {"min_price": {"$lte": price_val * (1 + PRICE_RANGE_MULTIPLIER)}}]}
    except Exception:
        pass
    return {"collection": target_collection, "where_filter": where_filter, "search_text": search_text.strip(),
            "search_type": search_type}


def get_best_product_match(client, query_details):
    collection_name = query_details["collection"]
    if not collection_name: return None, "Lütfen sorgunuzda bir ürün kategorisi belirtin."
    try:
        collection = client.get_collection(name=collection_name)
        n_results = SPECIAL_N_RESULTS if query_details['search_type'] != SEARCH_TYPE_DEFAULT else DEFAULT_N_RESULTS
        result = genai.embed_content(model=EMBEDDING_MODEL, content=[query_details["search_text"]],
                                     task_type="RETRIEVAL_QUERY")
        query_embedding = result['embedding']
        results = collection.query(query_embeddings=query_embedding, n_results=n_results,
                                   where=query_details["where_filter"] or None)
        if not results or not results.get('metadatas') or not results['metadatas'][0]:
            return None, "Bu kriterlere uygun ürün bulunamadı."
        candidates = results['metadatas'][0]
        search_type = query_details['search_type']
        if search_type == SEARCH_TYPE_FP:
            best_product = max((p for p in candidates if p.get('min_price', 0) > 0),
                               key=lambda p: len(p.get('features', '')) / p['min_price'], default=None)
            return (best_product, "Başarılı") if best_product else (None, "F/P için uygun ürün yok.")
        elif search_type in [SEARCH_TYPE_CHEAPEST, SEARCH_TYPE_EXPENSIVE]:
            valid_products = [p for p in candidates if p.get('min_price', 0) > 0]
            if not valid_products: return None, "Fiyat bilgisi olan ürün bulunamadı."
            return (
                sorted(valid_products, key=lambda p: p['min_price'], reverse=(search_type == SEARCH_TYPE_EXPENSIVE))[0],
                "Başarılı")
        else:
            return candidates[0], "Başarılı"
    except Exception as e:
        print(f"HATA: Ürün arama sırasında bir sorun oluştu: {e}")
        return None, "Arama sırasında beklenmedik bir sorun oluştu."


def generate_final_prompt(user_question, product_context, history):
    urun_adi = product_context.get('product_name', 'N/A')
    urun_linki = product_context.get('product_url', 'N/A')
    urun_ozellikleri = product_context.get('features', 'Mevcut değil.')
    try:
        offers = json.loads(product_context.get('offers_json', '[]'))
    except (json.JSONDecodeError, TypeError):
        offers = []
    satici_bilgisi_listesi, en_ucuz_satici_linki = [], urun_linki
    valid_offers = sorted([o for o in offers if o.get('price') is not None and o.get('price') > 0],
                          key=lambda x: x['price'])
    if valid_offers:
        en_ucuz_satici_linki = valid_offers[0].get('offer_url', urun_linki)
        for offer in valid_offers: satici_bilgisi_listesi.append(
            f"- {offer.get('seller_name', 'N/A')}: {offer.get('price', 'N/A')} TL")
    satici_bilgisi = "\n".join(satici_bilgisi_listesi) or "Online satıcı bilgisi bulunamadı."
    history_text = "\n".join([f"Kullanıcı: {h['user']}\nAsistan: {h['assistant']}" for h in history])

    return f"""GÖREV: Akıllı Satış Asistanı
[ROL TANIMLAMA (PERSONA)]
Sen, son derece bilgili, ikna edici, güvenilir ve proaktif bir "Akıllı Satış Asistanı"sın. Amacın, müşterilere sunduğun ürün hakkında en doğru bilgiyi vererek onları satın almaya teşvik etmek ve tüm sorularını profesyonel bir dille yanıtlamaktır.

[BAĞLAM (CONTEXT)]
1. Konuşma Geçmişi
Önceki diyalogları anlamak için bu bölümü dikkatle incele.
<konuşma_geçmişi>
{history_text}
</konuşma_geçmişi>

2. Sağlanan Ürün Bilgileri (Bilgi Kaynağı - RAG)
Yanıtlarını oluştururken temel alacağın tek ve yegane bilgi kaynağı burasıdır.
<ürün_bilgileri>
Ürün Adı: {urun_adi}
Ürün Özellikleri: {urun_ozellikleri}
Fiyat Bilgileri ve Satıcılar:
{satici_bilgisi}
En Uygun Fiyatlı Satıcı Linki: {en_ucuz_satici_linki}
</ürün_bilgileri>

[TALİMATLAR VE SÜREÇ (ADIM ADIM DÜŞÜNME)]
Müşterinin sorusuna yanıt vermeden önce aşağıdaki adımları sırasıyla takip et:
Adım 1: Analiz Et
Müşterinin son sorusunun ({user_question}) ne anlama geldiğini ve neyi amaçladığını anla.
Konuşma geçmişindeki ({history_text}) önceki konularla bağlantısını kur.
Soruyu yanıtlamak için hangi bilgilere ihtiyacın olduğunu belirle.
Adım 2: Bilgiyi Değerlendir
İhtiyaç duyduğun bilgilerin <ürün_bilgileri> kaynağında olup olmadığını kontrol et.
Eğer Bilgi Yeterliyse: Yanıtını sadece ve sadece <ürün_bilgileri> bölümündeki verileri kullanarak oluştur. KESİNLİKLE dışarıdan bilgi ekleme veya varsayımda bulunma.
Eğer Bilgi Yetersizse: Müşteriye, aradığı bilginin mevcut belgelerde olmadığını belirt. Araştırmaya başlamak için ilk olarak en uygun fiyatlı satıcının linkini ({en_ucuz_satici_linki}) kullan. Ardından, "Sizin için hızlıca bir araştırma yaptım ve şunları buldum:" diyerek bulduğun en alakalı ve güvenilir bilgileri özetleyerek sun.
Adım 3: Yanıtı Oluştur ve Sun
Ton ve Üslup: İkna edici, samimi, profesyonel ve yardımsever bir dil kullan. Karmaşık teknik detayları herkesin anlayabileceği şekilde basitleştir.
İçerik: Müşterinin sorusunu doğrudan yanıtla. Cevabında, ürünün özelliklerinin müşteriye sağlayacağı faydaları vurgula. Fiyat karşılaştırması yaparken en uygun seçeneği ve linkini öne çıkar.

[YENİ KULLANICI SORUSU]
{user_question}"""


# --- API ENDPOINT ---
@app.route('/chat', methods=['POST'])
def chat_handler():
    data = request.get_json()
    if not data or 'query' not in data or 'session_id' not in data:
        return Response(json.dumps({"error": "Geçersiz istek"}), status=400, mimetype='application/json')

    user_question = data['query']
    session_id = data['session_id']
    history = get_conversation_history(session_id)
    query_details = extract_query_details(user_question)

    if not query_details["collection"] and history:
        last_product_context = history[-1].get("product_context")
        if last_product_context:
            if query_details["search_type"] in [SEARCH_TYPE_CHEAPEST, SEARCH_TYPE_EXPENSIVE]:
                query_details["collection"] = sanitize_collection_name(last_product_context.get("subcategory"))
                query_details["search_text"] = f"{last_product_context.get('product_name', '')} {user_question}"
                last_price = last_product_context.get("min_price", -1)
                if last_price > 0:
                    query_details["where_filter"] = {"min_price": {"$lt": last_price}} if query_details[
                                                                                              "search_type"] == SEARCH_TYPE_CHEAPEST else {
                        "min_price": {"$gt": last_price}}
                product_context, status = get_best_product_match(CLIENT, query_details)
            else:
                product_context, status = last_product_context, "Başarılı (Geçmişten)"
        else:
            product_context, status = get_best_product_match(CLIENT, query_details)
    else:
        product_context, status = get_best_product_match(CLIENT, query_details)

    if not product_context:
        # Ürün bulunamadığında bile stream formatında cevap verelim
        def empty_stream():
            yield f'2:{{ "error": "Üzgünüm, bu isteğe uygun bir ürün bulamadım. (Sebep: {status})" }}\n'

        return Response(stream_with_context(empty_stream()), mimetype="text/plain; charset=utf-8")

    final_prompt = generate_final_prompt(user_question, product_context, history)

    # STREAMING İÇİN YENİ YAPI (AI SDK v3/v4 UYUMLU)
    def stream_response():
        try:
            # 1. Önce ürün bilgisini (data) gönder.
            # AI SDK formatı: '0:' prefix'i ile JSON verisi.
            yield f'0:{json.dumps(product_context)}\n'

            # 2. Sonra modelden gelen metin akışını gönder.
            response_stream = MODEL.generate_content(final_prompt, stream=True)
            full_response_text = ""
            for chunk in response_stream:
                if chunk.text:
                    # AI SDK formatı: '1:' prefix'i ile metin verisi (JSON olarak encode edilmiş).
                    full_response_text += chunk.text
                    yield f'1:{json.dumps(chunk.text)}\n'

            # 3. Konuşma geçmişini tam metinle kaydet
            history.append({
                "user": user_question,
                "assistant": full_response_text,
                "product_context": product_context
            })
            save_conversation_history(session_id, history)

        except Exception as e:
            print(f"HATA: Stream sırasında bir sorun oluştu: {e}")
            # AI SDK formatı: '2:' prefix'i ile hata JSON'u.
            yield f'2:{{ "error": "Modelden cevap alınırken bir sorun oluştu." }}\n'

    # mimetype 'text/plain' olmalı, AI SDK bunu bekler.
    return Response(stream_with_context(stream_response()), mimetype='text/plain; charset=utf-8')


# --- UYGULAMAYI BAŞLATMA ---
initialize_services()

if __name__ == "__main__":
    print(">>> __main__ bloğu çalıştırılıyor (YEREL GELİŞTİRME).")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
