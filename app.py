from pathlib import Path

import streamlit as st

from chatbot import CyberLawChatbot
from ingest import ingest_knowledge_base, is_knowledge_base_loaded


BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


st.set_page_config(
    page_title="CyberLaw AI — Indian IT Act & IPC Assistant",
    page_icon="⚖️",
    layout="centered",
)


def ensure_knowledge_base() -> None:
    """Create the ChromaDB vectorstore on first launch if it is missing or empty."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    if not is_knowledge_base_loaded():
        ingest_knowledge_base()


@st.cache_resource(show_spinner=False)
def load_chatbot() -> CyberLawChatbot:
    """Cache the chatbot so models and vector connections load only once."""
    return CyberLawChatbot()


# Auto-ingest the legal knowledge base before the chatbot is initialized.
ensure_knowledge_base()

with st.sidebar:
    st.title("⚖️ CyberLaw AI")
    st.write("Indian Cyber Law ke baare mein kuch bhi poochiye")

    response_language = st.selectbox(
        "Answer language",
        ["Hinglish", "English"],
        index=0,
        help="Hinglish mode Roman Hindi + English mein jawab deta hai.",
    )

    st.subheader("Aap kya pooch sakte ho?")
    st.markdown(
        """
- Hacking ki punishment IT Act mein kya hai?
- IPC Section 420 cyber fraud mein kaise apply hota hai?
- IT Act Section 66C identity theft ke baare mein kya kehta hai?
- Online threatening messages crime hai kya?
- Indian law mein cyber terrorism kya hota hai?
"""
    )

    if st.button("Clear Chat / Chat saaf karein", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with Gemini Flash + ChromaDB | Free & Open Source")


st.header("⚖️ CyberLaw AI Assistant")
st.subheader("Indian IT Act, IPC, BNS aur cybercrime laws ko simple language mein samjhein")


# Store chat messages in Streamlit session state so history survives reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.messages:
    welcome_message = (
        "Namaste! 👋 Main CyberLaw AI hoon, Indian Cyber Law ke liye aapka assistant. "
        "Aap IT Act 2000, IPC, BNS, DPDP Act, IT Rules, ya common cybercrime topics ke baare mein pooch sakte ho. "
        "Main simple Hinglish mein answer dunga aur exact section cite karunga."
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})


# Render all previous chat messages with Streamlit's native chat UI.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


chatbot = None
try:
    chatbot = load_chatbot()
except Exception as exc:
    st.error(str(exc))
    st.stop()


user_input = st.chat_input("Indian cyber law ke baare mein poochiye...")

if user_input:
    # Add the user message immediately, then retrieve and generate the assistant reply.
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Legal database search ho raha hai..."):
            assistant_response = chatbot.get_response(user_input, response_language)
        st.markdown(assistant_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
