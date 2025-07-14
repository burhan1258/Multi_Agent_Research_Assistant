from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_summary_prompt():
    """Get the prompt template for document summarization"""
    return ChatPromptTemplate.from_template("""
You are a helpful assistant. Summarize the following document clearly and accurately:
<context>
{context}
</context>
""")

def summarize_document(llm, documents):
    """
    Summarize the uploaded document(s)
    
    Args:
        llm: Language model instance
        documents: List of document chunks
        
    Returns:
        str: Document summary
    """
    summary_prompt = get_summary_prompt()
    chain = create_stuff_documents_chain(llm, summary_prompt)
    return chain.invoke({"context": documents})