from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

def get_translate_prompt():
    """Get the prompt template for translation"""
    return ChatPromptTemplate.from_template("""
Translate the following content into {language}, preserving meaning and academic tone:
{content}
""")

def translate_text(llm, content, language):
    """
    Translate content to specified language
    
    Args:
        llm: Language model instance
        content: Content to translate
        language: Target language
        
    Returns:
        str: Translated content
    """
    # Handle dictionary output
    if isinstance(content, dict):
        combined_text = "\n\n".join(str(v) for v in content.values())
    else:
        combined_text = str(content)
    
    translate_prompt = get_translate_prompt()
    translate_chain = LLMChain(llm=llm, prompt=translate_prompt)
    result = translate_chain.invoke({
        "language": language,
        "content": combined_text
    })
    return result