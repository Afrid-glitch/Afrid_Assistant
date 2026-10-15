
import streamlit as st
from langchain_core.messages import HumanMessage


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Profile Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1020;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #10172a;
        border-right: 1px solid #26324b;
    }

    .hero {
        padding: 30px;
        border: 1px solid #293650;
        border-radius: 20px;
        background: linear-gradient(135deg, #17233e, #111827);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 38px;
        color: #ffffff;
        margin-bottom: 10px;
    }

    .hero p {
        color: #aab7cf;
        font-size: 16px;
        line-height: 1.7;
    }

    .tag {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px 5px 4px 0;
        border: 1px solid #354565;
        border-radius: 20px;
        color: #c4d6ff;
        font-size: 12px;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #26324b;
        border-radius: 14px;
        padding: 12px;
        background-color: #111a2d;
        margin-bottom: 12px;
    }

    div[data-testid="stChatInput"] {
        border-color: #354565;
    }

    .section-label {
        color: #91a9d8;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. LOAD YOUR EXISTING LANGGRAPH CHATBOT
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading AI profile assistant...")
def load_chatbot():
    # Importing main.py initializes your existing
    # PDF retrieval pipeline and LangGraph workflow.
    from main import app as chatbot_graph
    return chatbot_graph


# --------------------------------------------------
# 4. SESSION STATE
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("##  Profile AI")
    st.caption("AI-powered professional assistant")

    st.divider()

    st.markdown("### About this project")
    st.write(
        "An intelligent RAG-based chatbot that answers about Afrid pasha"
         "documents, project details, and certifications."
    )

    st.markdown("### Technology stack")

    st.markdown(
        """
        <span class="tag">Python</span>
        <span class="tag">LangGraph</span>
        <span class="tag">LangChain</span>
        <span class="tag">Groq LLM</span>
        <span class="tag">FAISS</span>
        <span class="tag">Hugging Face</span>
        <span class="tag">Streamlit</span>
        <span class="tag">RAG</span>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Suggested topics")
    st.caption("Skills, projects, education, certifications, career goals")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.divider()
    st.caption("Built with Python and Generative AI")


# --------------------------------------------------
# 6. MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="section-label">PERSONAL AI ASSISTANT</div>
        <h1>Explore My Professional Profile</h1>
        <p>
            Have a question about my technical skills, AI projects,
            certifications, or career interests? Ask my AI assistant.
            It retrieves relevant information from my profile documents
            and generates answers using an LLM.
        </p>
        <span class="tag">Retrieval-Augmented Generation</span>
        <span class="tag">Document-Based Answers</span>
        <span class="tag">AI-Powered Search</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 7. WELCOME SCREEN AND SUGGESTED QUESTIONS
# --------------------------------------------------

if not st.session_state.chat_history:
    st.markdown("### What would you like to know?")
    st.write(
        "Choose a question below or type your own question "
        "in the chat box."
    )

    suggestions = [
        "Tell me about yourself",
        "What are your technical skills?",
        "Explain your AI projects",
        "Which certifications have you completed?",
        "What role are you looking for?",
        "What experience do you have with Python?",
    ]

    for start in range(0, len(suggestions), 2):
        cols = st.columns(2)

        for offset, col in enumerate(cols):
            index = start + offset

            if index < len(suggestions):
                question = suggestions[index]

                with col:
                    if st.button(
                        question,
                        key=f"suggestion_{index}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_prompt = question
                        st.rerun()


# --------------------------------------------------
# 8. DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 9. HANDLE USER QUESTIONS
# --------------------------------------------------

user_prompt = st.chat_input(
    "Ask about my skills, projects, certifications..."
)

# A suggested question can be used without typing it.
if st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


if user_prompt:
    # Display the visitor's question.
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Run your existing RAG + LangGraph pipeline.
    with st.chat_message("assistant"):
        with st.spinner("Searching profile documents..."):
            try:
                chatbot = load_chatbot()

                result = chatbot.invoke(
                    {
                        "messages": [
                            HumanMessage(content=user_prompt)
                        ]
                    }
                )

                answer = str(result["messages"][-1].content)

            except Exception as exc:
                print(f"Chatbot error: {exc}")
                answer = (
                    "Sorry, I couldn't process your question "
                    "right now. Please try again."
                )

        st.markdown(answer)

    # Save both messages for the current browser session.
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
