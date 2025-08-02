import chromadb
import json
import sys
import time
import re

# --- AYARLAR ---
INPUT_FILENAME = 'veriler_vektorlu.json'
DB_PATH = "urun_veritabani"  # Veritabanının kaydedileceği klasör
# ChromaDB'ye tek seferde gönderilecek maksimum kayıt sayısı.
DB_BATCH_SIZE = 4096


# --- YARDIMCI FONKSİYONLAR ---

def sanitize_collection_name(name):
    """
    Kategori adını, ChromaDB için geçerli bir koleksiyon adına dönüştürür.
    Örnek: "Ankastre Set" -> "ankastre_set"
    """
    if not name:
        return "diger_kategoriler"
    # Türkçe karakterleri Latin karakterlere çevir
    char_map = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
    }
    for tr_char, en_char in char_map.items():
        name = name.replace(tr_char, en_char)

    name = name.lower()
    # Boşlukları ve tireleri alt çizgiye dönüştür
    name = name.replace(' ', '_').replace('-', '_')
    # Geçersiz karakterleri kaldır
    name = re.sub(r'[^a-z0-9_]', '', name)
    # Başta veya sonda alt çizgi olmamasını sağla
    name = name.strip('_')
    # ChromaDB'nin kurallarına uyum sağla
    if len(name) < 3:
        name = f"{name}_koleksiyonu"
    if len(name) > 63:
        name = name[:63]
    return name if name else "diger_kategoriler"


def get_min_price(product):
    """Bir ürünün teklifleri arasından en düşük geçerli fiyatı bulur."""
    try:
        # Teklifler doğrudan product objesinden alınır
        offers = product.get('offers', [])
        if not offers:
            return float('inf')

        valid_prices = [
            offer.get('price') for offer in offers
            if isinstance(offer.get('price'), (int, float)) and offer.get('price') > 0
        ]

        return min(valid_prices) if valid_prices else float('inf')
    except (TypeError, ValueError):
        return float('inf')


def main():
    """Ana program fonksiyonu."""
    print("--- Hiyerarşik Vektör Veritabanı Oluşturma Betiği Başlatıldı ---")
    start_time = time.time()

    # BÖLÜM 1: VERİYİ YÜKLEME
    print(f"\n[BÖLÜM 1] '{INPUT_FILENAME}' dosyasından ürün verileri okunuyor...")
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            products = json.load(f)
        print(f"-> Başarılı: '{INPUT_FILENAME}' dosyasından {len(products)} kayıt okundu.")
    except FileNotFoundError:
        sys.exit(f"HATA: Girdi dosyası '{INPUT_FILENAME}' bulunamadı.")
    except json.JSONDecodeError:
        sys.exit(f"HATA: '{INPUT_FILENAME}' dosyası geçerli bir JSON formatında değil.")

    # BÖLÜM 2: ÜRÜNLERİ KATEGORİLERE GÖRE GRUPLAMA
    print("\n[BÖLÜM 2] Ürünler alt kategorilere göre gruplanıyor...")
    categorized_products = {}
    for product in products:
        if not product.get('product_id') or not product.get('embedding_vector'):
            continue

        subcategory = product.get('subcategory') or "Diğer"
        if subcategory not in categorized_products:
            categorized_products[subcategory] = []
        categorized_products[subcategory].append(product)

    print(f"-> {len(categorized_products)} adet benzersiz alt kategori bulundu.")

    # BÖLÜM 3: HER KATEGORİ İÇİN VERİTABANI OLUŞTURMA VE DOLDURMA
    print("\n[BÖLÜM 3] Her kategori için ayrı koleksiyonlar oluşturuluyor ve dolduruluyor...")
    client = chromadb.PersistentClient(path=DB_PATH)

    for category_name, product_list in categorized_products.items():
        collection_name = sanitize_collection_name(category_name)
        print(f"\n--- İşleniyor: '{category_name}' -> Koleksiyon: '{collection_name}' ({len(product_list)} ürün) ---")

        # Koleksiyonu temizle ve yeniden oluştur
        try:
            if collection_name in [c.name for c in client.list_collections()]:
                client.delete_collection(name=collection_name)
            collection = client.create_collection(name=collection_name)
        except Exception as e:
            print(f"HATA: '{collection_name}' koleksiyonu oluşturulamadı: {e}")
            continue

        # Ürünleri fiyata göre sırala
        sorted_products = sorted(product_list, key=get_min_price)

        # ChromaDB için listeleri hazırla
        ids_list = [p['product_id'].replace('.html', '') for p in sorted_products]
        embeddings_list = [p['embedding_vector'] for p in sorted_products]

        metadata_list = []
        for p in sorted_products:
            min_fiyat = get_min_price(p)
            metadata = {
                "product_name": p.get('product_name') or 'N/A',
                "product_url": f"https://www.akakce.com/p/-{p.get('product_id', '').replace('.html', '')}",
                "features": p.get('features') or 'N/A',
                "subcategory": p.get('subcategory') or 'N/A',
                # En düşük fiyatı meta veriye ekle (sonsuz ise -1 yap)
                "min_price": min_fiyat if min_fiyat != float('inf') else -1,
                "offers_json": json.dumps(p.get('offers', []))
            }
            metadata_list.append(metadata)

        # Verileri gruplar halinde veritabanına ekle
        total_items = len(ids_list)
        for i in range(0, total_items, DB_BATCH_SIZE):
            end_index = min(i + DB_BATCH_SIZE, total_items)

            batch_ids = ids_list[i:end_index]
            batch_embeddings = embeddings_list[i:end_index]
            batch_metadatas = metadata_list[i:end_index]

            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            print(f"-> {end_index} / {total_items} ürün '{collection_name}' koleksiyonuna eklendi...")

    end_time = time.time()
    print("\n--- Tüm işlemler başarıyla tamamlandı! ---")
    print(f"Toplam {len(categorized_products)} adet koleksiyon oluşturuldu/güncellendi.")
    print(f"Toplam süre: {int(end_time - start_time)} saniye")


if __name__ == "__main__":
    main()
