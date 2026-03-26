# Voice Genetics

An interactive application for **voice analysis and acoustic feature extraction**, combining a **Streamlit frontend** with a **FastAPI backend**.

👉 **Live app:**
[https://voice-genetics.streamlit.app](https://voice-genetics.streamlit.app)

---

## Features

### Streamlit App (User Interface)

* 🎤 Upload and analyze voice recordings directly in the browser
* 📊 Interactive visualization of:

  * Pitch (F0)
  * Formants
  * MFCCs
  * Voice quality (jitter, shimmer, HNR)
* Built-in explanations for each metric (UI-friendly)
* Download results as JSON or CSV
* History of recent analyses

---

### ⚙️ Backend API (FastAPI)

* 🚀 High-performance REST API
* 🎵 Audio processing with `librosa` and `parselmouth`
* 📊 Feature extraction using `numpy` and `scipy`
* 📁 File upload support
* 📚 Swagger UI documentation
* 🔄 Real-time feature extraction

---

## 🧠 Extracted Features

The system analyzes voice recordings and extracts:

* **Pitch (F0):** mean, min, max, variability
* **Voice quality:** jitter, shimmer, harmonic-to-noise ratio (HNR)
* **Timbre:**

  * Formants (F1–F4+)
  * MFCCs (spectral features)
* **Recording quality:**

  * Duration
  * Signal-to-noise ratio (SNR)
  * Background noise level

> ⚠️ These metrics are for analysis and research purposes only and are **not a medical diagnosis**.

---

## 🚀 Quick Start

### ▶️ Run Streamlit App (recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run streamlit_app.py
```

Then open:

```
http://localhost:8501
```

---

### ⚙️ Run FastAPI Backend (optional)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

* API: [http://localhost:8000](http://localhost:8000)
* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🐳 Using Docker

```bash
# Build image
docker build -t voice-genetics .

# Run container
docker run -d -p 8000:8000 --name voice-genetics voice-genetics
```

---

## API Endpoints

| Method | Endpoint         | Description                          |
| ------ | ---------------- | ------------------------------------ |
| GET    | `/health`        | Health check                         |
| POST   | `/extract`       | Extract features from one audio file |
| POST   | `/extract/batch` | Extract features from multiple files |
| GET    | `/config`        | Get current configuration            |
| POST   | `/config`        | Update extraction settings           |

---

## Dependencies

| Package           | Purpose                     |
| ----------------- | --------------------------- |
| streamlit         | UI framework                |
| fastapi           | Backend API                 |
| uvicorn           | ASGI server                 |
| numpy             | Numerical computing         |
| scipy             | Scientific computing        |
| librosa           | Audio analysis              |
| praat-parselmouth | Phonetic feature extraction |
| pandas            | Data handling               |
| soundfile         | Audio I/O                   |

---

## Project Structure

```
voice-genetics/
│
├── streamlit_app.py      # Streamlit frontend
├── main.py               # FastAPI backend
├── audio_processor.py    # Feature extraction logic
├── config.py             # Configuration
├── requirements.txt
└── Dockerfile
```

---

## 💡 Notes

* The **Streamlit app is the main interface** for users
* The **FastAPI backend can be used separately** for integration or batch processing
* For best compatibility, Python **3.10–3.11** is recommended

---

## Deployment

The app is deployed using **Streamlit Community Cloud**:

👉 [https://voice-genetics.streamlit.app](https://voice-genetics.streamlit.app)
