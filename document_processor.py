import PyPDF2
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def process_pdfs(uploaded_files):
    """
    Extract text from uploaded PDF files and split into chunks
    
    Args:
        uploaded_files: List of uploaded PDF files
        
    Returns:
        list: List of document chunks
    """
    documents = []
    for file in uploaded_files:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        documents.append(Document(page_content=text, metadata={"source": file.name}))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def create_vector_store(documents, embedding):
    """
    Create FAISS vector store from documents
    
    Args:
        documents: List of document chunks
        embedding: Embedding model
        
    Returns:
        FAISS: Vector store
    """
    return FAISS.from_documents(documents, embedding)