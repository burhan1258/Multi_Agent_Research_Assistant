from langchain.chains import RetrievalQA

def chat_with_paper(llm, vectorstore, query):
    """
    Chat with the paper using Q&A
    
    Args:
        llm: Language model instance
        vectorstore: FAISS vector store
        query: User's question
        
    Returns:
        str: Answer to the question
    """
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)