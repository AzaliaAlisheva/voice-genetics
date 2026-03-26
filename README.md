# 🎵 Voice Genetics

A FastAPI-based REST API for audio processing and analysis using parselmouth, librosa and numpy.


## ✨ Features

- 🚀 FastAPI framework for high-performance API
- 🎵 Audio processing with librosa
- 📊 NumPy for numerical operations
- 🐳 Docker support for easy deployment
- 📚 Automatic interactive API documentation (Swagger UI)
- 🔄 Real-time audio analysis
- 📁 File upload support for audio processing


## 🚀 Quick Start

### Using Python directly:

```bash
# Clone the repository
git clone https://github.com/AzaliaAlisheva/voice-genetics.git
cd voice-genetics

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker:

```bash
# Build the Docker image
docker build -t voice-genetics .

# Run the container
docker run -d -p 8000:8000 --name voice-genetics voice-genetics
```

Once running, open your browser at:

* API: http://localhost:8000
* Swagger UI: http://localhost:8000/docs


## 🔌 Endpoints

| Method | Endpoint       | Description | Parameters |
| ------ | -------------- | ----------- | ---------- |
| GET    | /health        | Health check endpoint | - |
| POST	 | /extract       | Extract acoustic features from uploaded audio file | file: Audio file (WAV, MP3, etc.) |
| POST	 | /extract/batch |	Extract features from multiple audio files | the list of files: Audio files (WAV, MP3, etc.) |
| GET    | /config        |	Get current configuration | - |
| POST   | /config        | Update feature extraction configuration | see the Swagger docs |

## 📦 Dependencies
| Package | Version | Purpose |
| ------- | ------- | ------- |
| fastapi | 0.104.1 | Web framework |
| uvicorn |	0.24.0	| ASGI server |
| numpy | 1.24.3 | Numerical computing |
| librosa | 0.10.1 | Audio analysis |
| scipy | 1.11.4 | Scientific computing |
| pydantic | 2.5.0 | Data validation |
| python-multipart | 0.0.18 | File upload support
| praat-parselmouth | 0.4.7 | Phonetic analysis (pitch, formants, intensity, etc.) |

