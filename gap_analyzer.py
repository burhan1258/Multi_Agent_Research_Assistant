from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from summarizer import get_summary_prompt

def get_gap_prompt():
    """Get the prompt template for research gap analysis"""
    return ChatPromptTemplate.from_template("""
Analyze the following summary and identify key research gaps, unanswered questions, or limitations:
{summary}
""")

def identify_research_gaps(llm, documents):
    """
    Identify research gaps in the document
    
    Args:
        llm: Language model instance
        documents: List of document chunks
        
    Returns:
        str: Research gaps analysis
    """
    # First get summary
    summary_prompt = get_summary_prompt()
    chain1 = create_stuff_documents_chain(llm, summary_prompt)
    summary = chain1.invoke({"context": documents})
    
    # Then analyze gaps
    gap_prompt = get_gap_prompt()
    chain2 = LLMChain(llm=llm, prompt=gap_prompt)
    return chain2.invoke({"summary": summary})