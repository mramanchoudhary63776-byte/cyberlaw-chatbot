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
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
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

        # Configure Gemini, then choose available Flash models for this API key.
        genai.configure(api_key=self.api_key)
        self.available_gemini_models = self.get_available_gemini_models()
        self.gemini_model_name = self.available_gemini_models[0]
        self.gemini_model = genai.GenerativeModel(self.gemini_model_name)

    def get_available_gemini_models(self) -> list[str]:
        """Return Gemini models that support generateContent for this account."""
        try:
            available_models = {
                model.name.replace("models/", "")
                for model in genai.list_models()
                if "generateContent" in getattr(model, "supported_generation_methods", [])
            }

            matching_models = [
                model_name for model_name in GEMINI_MODEL_CANDIDATES if model_name in available_models
            ]

            if matching_models:
                return matching_models

        except Exception:
            # If model listing fails, fall back to a lightweight Flash model.
            pass

        return ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]

    @staticmethod
    def is_model_retry_error(error: Exception) -> bool:
        """Return True when trying another Gemini model may solve the failure."""
        message = str(error).lower()
        return "429" in message or "quota" in message or "not found" in message

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

    @staticmethod
    def language_instruction(response_language: str) -> str:
        """Tell Gemini what language style to use for the final answer."""
        if response_language == "Hinglish":
            return (
                "Answer in simple Hinglish using Roman Hindi plus English. "
                "Use words like kanoon, saza, dhara, complaint, report, fraud, and section naturally. "
                "Keep legal section names, penalties, and citations in clear English. "
                "Do not use Devanagari script."
            )

        return "Answer in simple English."

    def build_prompt(self, query: str, context_sections: list, response_language: str = "Hinglish") -> str:
        """Build a grounded prompt using retrieved legal context."""
        context = "\n\n".join(context_sections) if context_sections else "No matching legal sections found."
        language_rule = self.language_instruction(response_language)

        return f"""
System: You are CyberLaw AI, an expert assistant on Indian Cyber Law, including the IT Act 2000, IPC, BNS 2023, DPDP Act 2023, IT Rules 2021, and practical cybercrime reporting topics. You only answer questions related to Indian cyber law, cybercrime, digital offences, data protection, and related criminal law sections. If a question is outside this domain, politely say you can only help with Indian cyber law topics. Always cite the specific section number or topic name in your answer. Be clear, helpful, and use simple English.

Language Style: {language_rule}

Context:
{context}

User Question:
{query}
""".strip()

    def get_response(self, query: str, response_language: str = "Hinglish") -> str:
        """Generate an answer with RAG context and Gemini Flash."""
        try:
            # Retrieval happens first, then Gemini answers using the grounded context.
            context_sections = self.get_relevant_sections(query)
            prompt = self.build_prompt(query, context_sections, response_language)
            last_error = None

            for model_name in self.available_gemini_models:
                try:
                    model = (
                        self.gemini_model
                        if model_name == self.gemini_model_name
                        else genai.GenerativeModel(model_name)
                    )
                    response = model.generate_content(prompt)

                    if response and getattr(response, "text", None):
                        self.gemini_model_name = model_name
                        self.gemini_model = model
                        return response.text.strip()

                except Exception as exc:
                    last_error = exc
                    if not self.is_model_retry_error(exc):
                        raise

            if last_error:
                raise last_error

            return "I could not generate a response right now. Please try asking again."

        except Exception as exc:
            return (
                "Sorry, I ran into a problem while answering. "
                "Please check your Gemini API key, internet connection, and local vectorstore, then try again. "
                f"Details: {exc}"
            )
