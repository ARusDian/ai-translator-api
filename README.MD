# 🌐 Indonesian Translation API

API penerjemah teks dan HTML dari Bahasa Indonesia ke Bahasa Inggris, Mandarin, Arab, dan Korea menggunakan model NLLB dan Helsinki-NLP, dengan caching Redis dan monitoring Prometheus + Grafana.

---

## 🚀 Fitur

- Translate teks biasa dan HTML dari `id` ke `en`, `zh`, `ar`, `ko`
- Caching otomatis menggunakan Redis dengan LRU eviction
- Monitoring Prometheus + Grafana (jumlah request, endpoint)
- Logging ke file dan Redis (UUID per request)
- Rate limiting per API key
- Dokumentasi Swagger otomatis (localhost:8000/docs)

---

## 🧱 Teknologi

- FastAPI + Uvicorn
- Transformers (HuggingFace)
- Redis (caching & rate limit)
- Prometheus + Grafana (monitoring)
- Docker + Docker Compose

---

## 🛠️ Instalasi

### 1. Clone Repo & Masuk Folder
```bash
git clone https://github.com/your-org/translator-api.git
cd translator-api
````

### 2. Buat `.env`

```env
TRANSLATION_API_KEY=your_api_key_here
REDIS_HOST=redis
REDIS_PORT=6379
```

### 3. Jalankan Docker Compose

```bash
docker-compose up -d --build
```

---

## 📦 Endpoint API

### 🔐 Autentikasi

Gunakan header berikut untuk semua request:

```http
Authorization: Bearer <TRANSLATION_API_KEY>
```

### `POST /translate`

Menerjemahkan teks biasa dari Bahasa Indonesia.

```json
POST /translate
{
  "target_lang": "ko",
  "text": "Saya suka belajar bahasa asing."
}
```

**Response:**

```json
{ "translated_text": "외국어를 배우는 게 정말 좋아요." }
```

---

### `POST /translate_html`

Menerjemahkan teks dalam HTML.

```json
POST /translate_html
{
  "target_lang": "zh",
  "html": "<p>Kucing itu lucu.</p>"
}
```

**Response:**

```json
{ "translated_html": "<p>那只猫很可爱。</p>" }
```

---

### `GET /logs/{api_key}`

Lihat log penggunaan per hari (opsional `?date=YYYY-MM-DD`)

### `GET /list_supported_languages`

Melihat daftar bahasa tujuan yang didukung.

### `GET /metrics`

Endpoint Prometheus (otomatis dibaca oleh Prometheus server)

---

## 📊 Akses Grafana

* URL: [http://localhost:3000](http://localhost:3000)
* Default user: `admin` / `admin`
* Tambahkan Prometheus sebagai data source ([http://prometheus:9090](http://prometheus:9090))

---

## 🔒 Keamanan Tambahan (Opsional)

* Gunakan NGINX untuk reverse proxy + HTTPS
* Batasi akses ke port 8000, 3000, 9090 hanya untuk internal
* Monitor rate limit dan abuse dari Redis logs

---

## 📄 Lisensi

MIT License. Dibuat oleh [Sentra Technology](https://sentratechnology.com).
