#!/bin/bash

# GCS'teki bucket adınız ve klasör adı
BUCKET_NAME="rag-api-veritabani" # KENDİ BUCKET ADINIZI YAZIN
SOURCE_FOLDER="urun_veritabani"
DESTINATION_FOLDER="./urun_veritabani"

# Hedef klasörün zaten var olup olmadığını kontrol et
if [ -d "$DESTINATION_FOLDER" ]; then
    echo "Veritabanı klasörü zaten mevcut, indirme atlanıyor."
else
    echo "Veritabanı GCS'ten indiriliyor..."
    # gcloud storage cp komutu ile tüm klasörü indir
    # -r: recursive (tüm alt dosyaları al), -n: no-clobber (varsa üzerine yazma)
    gcloud storage cp -r "gs://${BUCKET_NAME}/${SOURCE_FOLDER}" .
    echo "İndirme tamamlandı."
fi

# Gunicorn sunucusunu başlat
# Procfile'daki komutu buraya taşıdık
echo "Gunicorn sunucusu başlatılıyor..."
gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 API:app