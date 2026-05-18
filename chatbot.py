import os
from pathlib import Path

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "cyberlaw_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_CANDIDATES = [
    "gemini-1.5-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


class CyberLawChatbot:
    def __init__(self):
        """Initialize environment, vector search, local embeddings, and Gemini."""
        # Load secrets from .env without ever hardcoding API keys in source code.
        load_dotenv(BASE_DIR / ".env")

        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Copy .env.example to .env and add your Gemini API key."
            )

        # Connect to the local ChromaDB vectorstore created by ingest.py.
        self.client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Configure Gemini, then choose the first available Flash model for this API key.
        genai.configure(api_key=self.api_key)
        self.gemini_model_name = self.get_available_gemini_model()
        self.gemini_model = genai.GenerativeModel(self.gemini_model_name)

    def get_available_gemini_model(self) -> str:
        """Pick a Gemini model that supports generateContent for this account."""
        try:
            available_models = {
                model.name.replace("models/", "")
                for model in genai.list_models()
                if "generateContent" in getattr(model, "supported_generation_methods", [])
            }

            for model_name in GEMINI_MODEL_CANDIDATES:
                if model_name in available_models:
                    return model_name

        except Exception:
            # If model listing fails, fall back to the current stable Flash model.
            pass

        return "gemini-2.5-flash"

    def get_relevant_sections(self, query: str, top_k=3) -> list:
        """Find the most relevant legal sections for a user question."""
        # Convert the user's question into the same embedding space as the legal sections.
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents"],
        )

        documents = results.get("documents", [[]])
        return documents[0] if documents and documents[0] else []

    def build_prompt(self, query: str, context_sections: list) -> str:
        """Build a grounded prompt using retrieved legal context."""
        context = "\n\n".join(context_sections) if context_sections else "No matching legal sections found."

        return f"""
System: You are CyberLaw AI, an expert assistant on Indian Cyber Law (IT Act 2000 and IPC). You only answer questions related to Indian cyber law, cybercrime, digital offences, and related IPC sections. If a question is outside this domain, politely say you can only help with Indian cyber law topics. Always cite the specific section number in your answer. Be clear, helpful, and use simple English.

Context:
{context}

User Question:
{query}
""".strip()

    def get_response(self, query: str) -> str:
        """Generate an answer with RAG context and Gemini Flash."""
        try:
            # Retrieval happens first, then Gemini answers using the grounded context.
            context_sections = self.get_relevant_sections(query)
            prompt = self.build_prompt(query, context_sections)
            response = self.gemini_model.generate_content(prompt)

            if response and getattr(response, "text", None):
                return response.text.strip()

            return "I could not generate a response right now. Please try asking again."

        except Exception as exc:
            return (
                "Sorry, I ran into a problem while answering. "
                "Please check your Gemini API key, internet connection, and local vectorstore, then try again. "
                f"Details: {exc}"
            )
