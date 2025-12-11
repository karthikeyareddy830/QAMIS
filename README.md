# QAMIS

Quantum Authenticated Media Integrity System — Phase 1
(Short description: API + Streamlit dashboard to ingest images, sign and verify, and perform lightweight forensic checks.)
# ⭐ QAMIS — Quantum Authenticated Media Integrity System  
*A next-generation authenticity verification framework combining Quantum Security + AI Deepfake Forensics.*

---

## 📌 Overview

With generative AI becoming extremely advanced, forged images, deepfakes, and modified documents can be created in seconds.  
Traditional metadata or watermark-based validation is no longer reliable.

**QAMIS** (Quantum Authenticated Media Integrity System) provides a secure pipeline to validate whether an image is *original, altered, or forged*, using:

- Quantum-inspired digital signatures (Ed25519, PQC-ready design)  
- AI-powered media forensic analysis (pHash, SSIM, noise metrics)  
- A real-time visual dashboard (Streamlit)

---

## 🚀 Key Features

### ✔ Secure Image Ingestion
- Stores original image  
- Generates SHA-256 digest  
- Computes perceptual hash (pHash)  
- Creates a digital signature (Ed25519)

### ✔ Verification Engine
- Confirms authenticity using stored signature  
- Detects tampering & manipulated regions  
- Provides decision classification

### ✔ AI Forensic Metrics
- SSIM (structural similarity)  
- Noise inconsistency measurement  
- pHash distance  
- Combined authenticity score (0–100)

### ✔ Streamlit Dashboard
- Upload images  
- View forensic breakdown  
- Visual difference heatmaps  
- Side-by-side comparison  
- Authenticity score gauge meter  

---

## 🏗 System Architecture

         ┌──────────────────────────┐
         │       Streamlit UI        │
         │ (Dashboard & Visualization)│
         └───────────────┬───────────┘
                         │ REST API
                         ▼
            ┌──────────────────────────┐
            │       FastAPI Backend     │
            │  /ingest /verify /analyze │
            └──────────────┬───────────┘
                           │
           ┌───────────────┴────────────────┐
           │   AI Forensics Engine (Python)  │
           │  - pHash                        │
           │  - SSIM                         │
           │  - Noise Analysis               │
           └──────────────┬─────────────────┘
                           │
                ┌──────────┴────────────┐
                │  Cryptographic Layer   │
                │ - Ed25519 Signatures   │
                │ - PQC Ready Framework  │
                └──────────┬────────────┘
                           │
                 ┌─────────┴──────────┐
                 │  Storage + Metadata │
                 │   (Files + TinyDB)  │
                 └─────────────────────┘

---

## 🧠 AI Forensics Explained

| Metric | Meaning | Interpretation |
|--------|---------|----------------|
| **pHash Distance** | Structural similarity | Larger distance → More tampering |
| **SSIM** | Pixel-level similarity | < 0.9 indicates strong change |
| **Noise Metric** | Noise pattern mismatch | Indicates edits/filters |
| **Authenticity Score** | Weighted score (0–100) | 0 = fake, 100 = authentic |

---

## 📂 Folder Structure

QAMIS/
│── main.py # FastAPI backend
│── streamlit_app.py # Streamlit dashboard UI
│── metadata.json # TinyDB metadata storage
│── storage/ # Stored image files
│── keys/ # Ed25519 keypair
│── venv/ # Virtual environment (ignored in git)
│── requirements.txt # Dependencies
└── README.md # Documentation file


---

## 🛠 Tech Stack

**Backend:** FastAPI  
**Frontend:** Streamlit  
**Forensics:** NumPy, scikit-image, PIL  
**Crypto:** Ed25519 (PQC roadmap: Dilithium, Kyber)  
**DB:** TinyDB  
**Visualization:** Matplotlib, Plotly  

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/QAMIS.git
cd QAMIS
2. Create & activate virtual environment

Windows:

python -m venv venv
venv\Scripts\activate


Mac/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Start the FastAPI backend
uvicorn main:app --reload


Backend available at:

http://127.0.0.1:8000/docs

5. Start the Streamlit dashboard
streamlit run streamlit_app.py


Dashboard available at:

http://localhost:8501

🖥 Usage Guide
1. Ingest a genuine/original image

Upload image

System stores + signs it

Shows unique ID (SHA-256)

2. Analyze a new image

Upload second version

Backend calculates:

pHash distance

SSIM

Noise metric

Authenticity score

3. Visual comparison

Side-by-side original vs uploaded

Amplified difference heatmap

Percentage of changed pixels

4. Verification

Confirms whether signature + data match

Outputs classification:

Authentic

Probably authentic

Suspicious

Tampered

🔮 Future Enhancements (Roadmap)
Phase 2 — Deepfake & AI Manipulation Detection

GAN fingerprinting

Photoshop edit detection

Face morph detection

Phase 3 — Quantum Security Upgrade

Migrate to CRYSTALS-Dilithium signatures

Metadata encryption using Kyber

Optional QKD simulator

Phase 4 — Deployment

Deploy backend on Render / AWS

Streamlit Cloud hosting

Automated CI/CD pipeline
