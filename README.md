## Prasyarat

**Sebelum menjalankan, pastikan:**
- Go minimal versi 1.20
- Qdrant berjalan secara lokal di http://localhost:6333

Jika belum punya Qdrant, bisa jalankan via Docker:

```sh
docker run -p 6333:6333 qdrant/qdrant
```

**Menjalankan Program**

Clone lalu jalankan:

```sh
go run main.go
```

**Program akan otomatis:**

- Membuat collection articles (vector dim=4)
- Menyimpan 3 artikel contoh
- Melakukan pencarian vektor [0.1, 0.2, 0.3, 0.4]
- Menampilkan artikel yang paling mirip
- Membuat collection articles_embeddings (dim=128)
- Menghasilkan 1000 embedding acak
- Melakukan BulkInsert concurrent dengan batch size 100
- Menampilkan jumlah poin yang berhasil disimpan.