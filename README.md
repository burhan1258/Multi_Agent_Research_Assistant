🤖 Multi-Agent Research Assistant

An intelligent multi-agent AI system designed to assist researchers, students, and professionals in extracting key insights from research papers, identifying research gaps, generating ideas, simulating academic debates, visualizing insights, and interacting conversationally with documents all in one unified app.

🚨 Problem Statement

Research papers are dense, time-consuming to digest, and difficult to analyze efficiently. Researchers often need to summarize large texts, find gaps, generate new ideas, or translate results into different languages — all using multiple tools.

💡 Our Solution

We built an AI-powered multi-agent research assistant that automates this workflow. With a master agent that orchestrates multiple sub-agents — each specialized in a cognitive task users can perform end-to-end research analysis in a single interface.
The task is provided in plain language, and intelligent agents collaborate to complete it.

🧠 Key Features (Agents)

1. 📝 Summarization Agent – Summarizes academic papers clearly and concisely.
2. 🔍 Gap Analyzer Agent – Identifies research gaps and unanswered questions.
3. 💡 Idea Generator Agent – Suggests new research project ideas based on gaps.
4. 🎭 Debate Simulator Agent – Simulates a conversation between a critic and a supporter.
5. 📎 Citation Agent – Generates accurate APA-style citations.
6. 🌐 Translator Agent – Translates AI output into multiple user-defined languages.
7. 📊 Visual Insight Agent – Extracts charts and tables from PDFs and generates visual summaries.
8. 💬 Semantic Q&A Agent – Allows users to chat with the paper and ask questions.
9. 📈 RLHF-style Feedback Agent (planned) – Learns from user feedback to improve responses.

🛠 Tech Stack

💬 LLM: [Groq API](https://groq.com/) (LLaMA3)  
🧠 Framework: LangChain
🎯 Vector Store: FAISS
🔎 Embeddings: HuggingFace (MiniLM)
📚 PDF Processing: PyPDF2, pdfplumber
📊 Visualization: Plotly, Matplotlib
🌐 Web UI: Streamlit
🌍 Deployment: Hugging Face Spaces / GitHub
⚙️ Components Breakdown

| Component               | Description                                     |
|------------------------|-------------------------------------------------|
| document_processor.py  | Loads and splits PDFs, builds vector store     |
| summarizer.py          | Summarizes documents using LLMs                 |
| gap_analyzer.py        | Identifies research gaps                        |
| idea_generator.py      | Suggests new research directions                |
| debate_simulator.py    | Simulates a scholarly debate                    |
| citation_generator.py  | Generates citations in APA style                |
| chat_handler.py        | Handles semantic Q&A over vector embeddings     |
| translator.py          | Translates agent responses                      |
| visualization.py       | Extracts and visualizes numerical info from PDFs|

🔍 Use Cases

- Academic research analysis
- PhD literature review support
- Research paper critique & brainstorming
- Multi-lingual knowledge sharing
- Visual data summarization from academic reports
- Educational assistant for students

🚀 Future Work

🧠 Feedback Learning Agent (RLHF-style)
🔗 Integration with Arxiv or Semantic Scholar APIs
🧾 Citation style selection (APA, MLA, Chicago, etc.)
🤝 Collaborative chat between multiple users
🧪 Plug-and-play agent module system

🌐 Live Demo

👉 Try it on Hugging Face Spaces:  
[https://huggingface.co/spaces/your-space-url](https://huggingface.co/spaces/chburhan64/PDF_Agent)

🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/burhan1258/Multi_Agent_Research_Assistant.git
cd Multi_Agent_Research_Assistant
pip install -r requirements.txt
streamlit run app.py
