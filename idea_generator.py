from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from summarizer import get_summary_prompt
from gap_analyzer import get_gap_prompt

def get_idea_prompt():
    """Get the prompt template for research idea generation"""
    return ChatPromptTemplate.from_template("""
Given the research gaps:
{gaps}
Suggest 2-3 original research project ideas or questions that address these gaps. Explain why they are valuable.
""")

def suggest_research_ideas(llm, documents):
    """
    Suggest research ideas based on identified gaps
    
    Args:
        llm: Language model instance
        documents: List of document chunks
        
    Returns:
        str: Research ideas suggestions
    """
    # First get summary
    summary_prompt = get_summary_prompt()
    chain1 = create_stuff_documents_chain(llm, summary_prompt)
    summary = chain1.invoke({"context": documents})
    
    # Then identify gaps
    gap_prompt = get_gap_prompt()
    chain2 = LLMChain(llm=llm, prompt=gap_prompt)
    gaps = chain2.invoke({"summary": summary})
    
    # Finally suggest ideas
    idea_prompt = get_idea_prompt()
    chain3 = LLMChain(llm=llm, prompt=idea_prompt)
    return chain3.invoke({"gaps": gaps})