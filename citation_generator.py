from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_citation_prompt():
    """Get the prompt template for citation generation"""
    return ChatPromptTemplate.from_template("""
Generate an APA-style citation based on the document content:
<context>
{context}
</context>
""")

def generate_citation(llm, documents):
    """
    Generate APA-style citation for the document
    
    Args:
        llm: Language model instance
        documents: List of document chunks
        
    Returns:
        str: APA citation
    """
    citation_prompt = get_citation_prompt()
    citation_chain = create_stuff_documents_chain(llm, citation_prompt)
    return citation_chain.invoke({"context": documents})