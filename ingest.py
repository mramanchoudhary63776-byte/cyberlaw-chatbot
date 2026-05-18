import argparse
import re
import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "cyber_law.txt"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "cyberlaw_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_knowledge_base() -> str:
    """Read the plain-text cyber law knowledge base."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {DATA_FILE}")
    return DATA_FILE.read_text(encoding="utf-8")


def split_sections(text: str) -> list[str]:
    """Split content into one chunk per [SECTION: ...] marker."""
    chunks = [chunk.strip() for chunk in re.split(r"(?=\[SECTION:\s*)", text) if chunk.strip()]
    return [chunk for chunk in chunks if chunk.startswith("[SECTION:")]


def section_id(section_text: str, fallback_index: int) -> str:
    """Create a stable ChromaDB document id from the section marker."""
    match = re.match(r"\[SECTION:\s*(.*?)\]", section_text)
    raw_id = match.group(1) if match else f"section-{fallback_index}"
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", raw_id).strip("-").lower()
    return normalized or f"section-{fallback_index}"


def section_name(section_text: str) -> str:
    """Extract the readable section name for metadata and logs."""
    match = re.match(r"\[SECTION:\s*(.*?)\]", section_text)
    return match.group(1) if match else "Unknown Section"


def get_collection():
    """Connect to the local persistent ChromaDB collection."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except ValueError as exc:
        if "tenant" not in str(exc).lower():
            raise
        print("Local ChromaDB store is invalid; recreating vectorstore")
        clear_vectorstore_files()
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        return client.get_or_create_collection(name=COLLECTION_NAME)


def clear_vectorstore_files() -> None:
    """Safely remove generated ChromaDB files inside this project only."""
    base_path = BASE_DIR.resolve()
    vectorstore_path = VECTORSTORE_DIR.resolve()

    if vectorstore_path == base_path or base_path not in vectorstore_path.parents:
        raise RuntimeError(f"Refusing to clear unsafe vectorstore path: {vectorstore_path}")

    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


def reset_collection() -> None:
    """Clear generated ChromaDB files so edited legal data can be embedded again."""
    clear_vectorstore_files()
    print("Existing knowledge base cleared")


def is_knowledge_base_loaded() -> bool:
    """Return True when the ChromaDB collection already has stored sections."""
    collection = get_collection()
    return collection.count() > 0


def ingest_knowledge_base() -> None:
    """Embed legal sections locally and store them in ChromaDB."""
    collection = get_collection()

    # Keep ingestion idempotent so repeated app launches do not duplicate data.
    if collection.count() > 0:
        print("Knowledge base already loaded")
        return

    raw_text = load_knowledge_base()
    sections = split_sections(raw_text)

    if not sections:
        raise ValueError("No legal sections found in data/cyber_law.txt")

    # Embeddings are generated locally with sentence-transformers; no API key is needed.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(sections, normalize_embeddings=True).tolist()

    ids = [section_id(section, index) for index, section in enumerate(sections, start=1)]
    metadatas = [{"section": section_name(section)} for section in sections]

    # Store each legal section as a retrievable document chunk in ChromaDB.
    collection.add(
        ids=ids,
        documents=sections,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    for metadata in metadatas:
        print(f"Stored: {metadata['section']}")

    print(f"Knowledge base loaded with {len(sections)} sections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CyberLaw knowledge base into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear and rebuild the ChromaDB knowledge base from data/cyber_law.txt",
    )
    args = parser.parse_args()

    if args.reset:
        reset_collection()

    ingest_knowledge_base()
