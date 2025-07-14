from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from summarizer import get_summary_prompt

def get_debate_prompt():
    """Get the prompt template for debate simulation"""
    return ChatPromptTemplate.from_template("""
Act as two researchers discussing a paper.
Supporter: Defends the core idea of the document.
Critic: Challenges its assumptions, methods, or impact.
Use the following summary as reference:
{summary}
Generate a short conversation between them.
""")

def simulate_debate(llm, documents):
    """
    Simulate a debate about the document
    
    Args:
        llm: Language model instance
        documents: List of document chunks
        
    Returns:
        str: Debate conversation
    """
    # First get summary
    summary_prompt = get_summary_prompt()
    chain = create_stuff_documents_chain(llm, summary_prompt)
    summary = chain.invoke({"context": documents})
    
    # Then simulate debate
    debate_prompt = get_debate_prompt()
    debate_chain = LLMChain(llm=llm, prompt=debate_prompt)
    return debate_chain.invoke({"summary": summary})