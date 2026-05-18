# CyberLaw AI Chatbot - Indian IT Act & IPC

## 1. Project Overview

CyberLaw AI Chatbot is a Streamlit web app that answers questions about Indian cyber law using Gemini Flash and a local ChromaDB knowledge base. It retrieves relevant IT Act, IPC, BNS, DPDP Act, IT Rules, and practical cybercrime topic entries, then explains them in plain English with section citations.

This project uses only free services and open-source libraries.

## 2. Features

- Streamlit chat interface with persistent session history
- Gemini `gemini-1.5-flash` response generation
- Local RAG pipeline with ChromaDB
- Local embeddings using `all-MiniLM-L6-v2`
- Auto-ingestion on first app launch
- Expanded knowledge base covering IT Act, IPC, BNS, DPDP Act, IT Rules, and common cybercrime topics
- Ready for free deployment on Hugging Face Spaces

## 3. Prerequisites

- Python 3.10+
- pip
- Free Gemini API key

## 4. Setup Instructions

```bash
git clone https://github.com/your-username/cyberlaw-chatbot.git
cd cyberlaw-chatbot
pip install -r requirements.txt
```

Copy the environment template and add your Gemini API key:

```bash
cp .env.example .env
```

On Windows PowerShell, you can use:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace `your_gemini_api_key_here` with your actual key.

Load the knowledge base:

```bash
python ingest.py
```

If you edit `data/cyber_law.txt` later, rebuild the local ChromaDB store:

```bash
python ingest.py --reset
```

Run the app:

```bash
streamlit run app.py
```

## 5. How to Get a Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Create a new API key.
4. Paste it into your `.env` file as `GEMINI_API_KEY`.

## 6. Project Structure

```text
cyberlaw-chatbot/
├── app.py
├── chatbot.py
├── ingest.py
├── data/
│   └── cyber_law.txt
├── vectorstore/
├── .env.example
├── requirements.txt
└── README.md
```

## 7. Tech Stack

| Layer | Tool | Cost |
| --- | --- | --- |
| Language | Python 3.10+ | Free |
| UI | Streamlit | Free |
| AI Brain | Google Gemini API `gemini-1.5-flash` | Free tier |
| Vector Store | ChromaDB local persistent store | Free |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free, local |
| Parsing | PyMuPDF | Free |
| Environment | python-dotenv | Free |
| Deployment | Hugging Face Spaces | Free tier |

## 8. Free Deployment on Hugging Face Spaces

1. Create a new Space at [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose Streamlit as the Space SDK.
3. Upload this project or connect a Git repository.
4. Add `GEMINI_API_KEY` in Space settings under Repository secrets.
5. Hugging Face will install `requirements.txt` and run the Streamlit app.

The app automatically creates the local ChromaDB vectorstore on first launch if it is missing.

## 9. Disclaimer

This chatbot is for educational purposes only. Not a substitute for legal advice.
