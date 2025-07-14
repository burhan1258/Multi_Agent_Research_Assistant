import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import all modules
from document_processor import process_pdfs, create_vector_store
from summarizer import summarize_document
from gap_analyzer import identify_research_gaps
from idea_generator import suggest_research_ideas
from debate_simulator import simulate_debate
from citation_generator import generate_citation
from chat_handler import chat_with_paper
from translator import translate_text

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")
st.title("🤖 Multi-Agent Research Assistant")
st.markdown("Enhance your research process with intelligent summarization, critique, debate, translation, citation, and interactive Q&A. Upload a research paper and let our agents do the thinking!")

# Load Groq LLM (Llama3)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Load embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# File uploader
uploaded_files = st.file_uploader("📁 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("📚 Process Documents"):
    with st.spinner("Processing documents and generating vector store..."):
        documents = process_pdfs(uploaded_files)
        st.session_state.documents = documents
        st.session_state.vectorstore = create_vector_store(documents, embedding)
    st.success("✅ Document vector store created!")

# Agent Activation
if "documents" in st.session_state:
    st.subheader("🎓 Master Agent: What would you like me to do?")
    task = st.selectbox("Choose a task:", [
        "Summarize document",
        "Identify research gaps",
        "Suggest research ideas",
        "Simulate a debate",
        "Generate citation",
        "Chat with paper"
    ])

    # Handle Chat with paper separately
    if task == "Chat with paper":
        query = st.text_input("💬 Ask a question about the paper:")
        if query and st.button("🚀 Ask Question"):
            with st.spinner("Searching paper for answer..."):
                output = chat_with_paper(llm, st.session_state.vectorstore, query)
                st.session_state["last_agent_output"] = output
    
    # Handle other tasks
    elif st.button("🚀 Run Agent"):
        with st.spinner("Running agents..."):
            docs = st.session_state.documents[:10]
            output = ""

            if task == "Summarize document":
                output = summarize_document(llm, docs)

            elif task == "Identify research gaps":
                output = identify_research_gaps(llm, docs)

            elif task == "Suggest research ideas":
                output = suggest_research_ideas(llm, docs)

            elif task == "Simulate a debate":
                output = simulate_debate(llm, docs)

            elif task == "Generate citation":
                output = generate_citation(llm, docs)

            if output:
                st.session_state["last_agent_output"] = output

# Final Display Section with Translation Option
if "last_agent_output" in st.session_state:
    output = st.session_state["last_agent_output"]

    translate_toggle = st.toggle("🌍 Translate the response?")

    if not translate_toggle:
        st.markdown("### 🤖 Agent Response")
        st.write(output)

    if translate_toggle:
        default_languages = ["Spanish", "French", "German", "Chinese", "Urdu", "Other"]
        selected_language = st.selectbox("Choose translation language:", default_languages)
        if selected_language == "Other":
            user_language = st.text_input("Please enter your desired language:", key="custom_lang")
        else:
            user_language = selected_language

        if user_language:
            translated = translate_text(llm, output, user_language)
            st.markdown(f"### 🌐 Translated Response ({user_language})")
            st.write(translated)