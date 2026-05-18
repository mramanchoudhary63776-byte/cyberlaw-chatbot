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
    st.write("Ask anything about Indian Cyber Law")

    st.subheader("What can I ask?")
    st.markdown(
        """
- What is punishment for hacking under IT Act?
- Explain IPC Section 420 in cyber context
- What does IT Act Section 66C say about identity theft?
- Is sending threatening messages online a crime?
- What is cyber terrorism under Indian law?
"""
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with Gemini Flash + ChromaDB | Free & Open Source")


st.header("⚖️ CyberLaw AI Assistant")
st.subheader("Your guide to Indian IT Act 2000 & IPC cyber provisions")


# Store chat messages in Streamlit session state so history survives reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.messages:
    welcome_message = (
        "Namaste! 👋 I'm CyberLaw AI, your assistant for Indian Cyber Law. "
        "Ask me about the IT Act 2000, its amendments, or related IPC sections. "
        "I'll explain the law in plain English and cite the exact section for you."
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


user_input = st.chat_input("Ask about Indian cyber law...")

if user_input:
    # Add the user message immediately, then retrieve and generate the assistant reply.
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal database..."):
            assistant_response = chatbot.get_response(user_input)
        st.markdown(assistant_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
