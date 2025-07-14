import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
import pdfplumber
import io

def get_data_extraction_prompt():
    """Get the prompt template for data extraction"""
    return ChatPromptTemplate.from_template("""
Analyze the following text and extract all numerical data, statistics, and measurements. 
Focus on:
- Tables with numerical values
- Statistical results (percentages, counts, means, etc.)
- Experimental data and measurements
- Survey results and responses
- Performance metrics and comparisons

Text to analyze:
{text}

Please extract the data in a structured format and identify what each number represents.
""")

def get_chart_analysis_prompt():
    """Get the prompt template for chart analysis"""
    return ChatPromptTemplate.from_template("""
Analyze the following data visualization and provide insights:

Data Summary: {data_summary}
Chart Type: {chart_type}

Please provide:
1. Key trends and patterns visible in the data
2. Statistical significance or notable findings
3. Implications for the research
4. Any surprising or important insights

Keep the analysis concise but informative.
""")

def extract_numerical_data_from_pdf(uploaded_files):
    """
    Extract numerical data from PDF files using pdfplumber
    
    Args:
        uploaded_files: List of uploaded PDF files
        
    Returns:
        dict: Extracted numerical data and text
    """
    extracted_data = {
        'tables': [],
        'numerical_text': [],
        'raw_text': ''
    }
    
    for file in uploaded_files:
        # Reset file pointer
        file.seek(0)
        
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            full_text = ""
            
            for page in pdf.pages:
                # Extract text
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        # Convert table to DataFrame if it has data
                        if table and len(table) > 1:
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                extracted_data['tables'].append(df)
                            except:
                                pass
            
            extracted_data['raw_text'] = full_text
    
    return extracted_data

def extract_numbers_with_regex(text):
    """
    Extract numerical patterns from text using regex
    
    Args:
        text: Text to analyze
        
    Returns:
        list: List of found numerical patterns
    """
    patterns = [
        r'\b\d+\.?\d*%',  # Percentages
        r'\b\d+\.?\d*\s*(?:participants|subjects|samples|cases)',  # Sample sizes
        r'p\s*[<>=]\s*\d+\.?\d*',  # P-values
        r'\b\d+\.?\d*\s*±\s*\d+\.?\d*',  # Mean ± SD
        r'\$\d+\.?\d*[MBK]?',  # Currency
        r'\b\d{4}\b',  # Years
        r'\b\d+\.?\d*\s*(?:kg|g|cm|m|mm|seconds?|minutes?|hours?|days?)',  # Units
    ]
    
    found_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_numbers.extend(matches)
    
    return found_numbers

def create_sample_charts(extracted_data):
    """
    Create sample charts from extracted data
    
    Args:
        extracted_data: Dictionary containing extracted data
        
    Returns:
        tuple: (matplotlib figure, plotly figure, data summary)
    """
    # Try to create charts from tables first
    if extracted_data['tables']:
        df = extracted_data['tables'][0]  # Use first table
        
        # Find numerical columns
        numerical_cols = []
        for col in df.columns:
            try:
                # Try to convert to numeric
                pd.to_numeric(df[col], errors='coerce')
                if not df[col].isna().all():
                    numerical_cols.append(col)
            except:
                pass
        
        if len(numerical_cols) >= 1:
            # Create bar chart with plotly
            fig_plotly = px.bar(
                df, 
                x=df.columns[0],
                y=numerical_cols[0] if numerical_cols else df.columns[1],
                title="Data from Research Paper",
                color_discrete_sequence=['#1f77b4']
            )
            
            # Create matplotlib chart
            fig_mpl, ax = plt.subplots(figsize=(10, 6))
            if len(df) <= 20:  # Only plot if reasonable number of rows
                ax.bar(range(len(df)), pd.to_numeric(df[numerical_cols[0]], errors='coerce').fillna(0))
                ax.set_title('Extracted Data Visualization')
                ax.set_xlabel('Data Points')
                ax.set_ylabel('Values')
            else:
                ax.text(0.5, 0.5, 'Too many data points to display', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Data Available (Too Large to Display)')
            
            data_summary = f"Extracted table with {len(df)} rows and {len(df.columns)} columns. Numerical columns: {numerical_cols}"
            
            return fig_mpl, fig_plotly, data_summary
    
    # If no tables, try to create chart from regex-extracted numbers
    numbers = extract_numbers_with_regex(extracted_data['raw_text'])
    
    if numbers:
        # Extract just percentages for a simple chart
        percentages = [float(re.findall(r'\d+\.?\d*', num)[0]) for num in numbers if '%' in num]
        
        if len(percentages) >= 2:
            # Create simple bar chart
            fig_mpl, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(percentages[:10])), percentages[:10])
            ax.set_title('Extracted Percentages from Text')
            ax.set_xlabel('Data Points')
            ax.set_ylabel('Percentage (%)')
            
            # Plotly version
            fig_plotly = px.bar(
                x=list(range(len(percentages[:10]))),
                y=percentages[:10],
                title="Extracted Percentages from Text",
                labels={'x': 'Data Points', 'y': 'Percentage (%)'}
            )
            
            data_summary = f"Extracted {len(percentages)} percentage values from text. Showing first 10."
            
            return fig_mpl, fig_plotly, data_summary
    
    # Default case - create a simple info chart
    fig_mpl, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No suitable numerical data found for visualization\nTry uploading a PDF with tables or statistical data', 
           transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_title('Data Extraction Result')
    ax.axis('off')
    
    fig_plotly = go.Figure()
    fig_plotly.add_annotation(
        text="No suitable numerical data found for visualization<br>Try uploading a PDF with tables or statistical data",
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font=dict(size=16)
    )
    fig_plotly.update_layout(title="Data Extraction Result")
    
    data_summary = "No suitable numerical data found in the document for visualization."
    
    return fig_mpl, fig_plotly, data_summary

def generate_visual_insights(llm, uploaded_files):
    """
    Generate visual insights from PDF data
    
    Args:
        llm: Language model instance
        uploaded_files: List of uploaded PDF files
        
    Returns:
        dict: Contains charts and AI analysis
    """
    try:
        # Extract data from PDF
        extracted_data = extract_numerical_data_from_pdf(uploaded_files)
        
        # Create charts
        fig_mpl, fig_plotly, data_summary = create_sample_charts(extracted_data)
        
        # Generate AI analysis
        analysis_prompt = get_chart_analysis_prompt()
        analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
        
        ai_analysis = analysis_chain.invoke({
            "data_summary": data_summary,
            "chart_type": "Bar Chart/Data Visualization"
        })
        
        return {
            'matplotlib_fig': fig_mpl,
            'plotly_fig': fig_plotly,
            'data_summary': data_summary,
            'ai_analysis': ai_analysis,
            'extracted_numbers': extract_numbers_with_regex(extracted_data['raw_text'][:2000]),  # First 2000 chars
            'tables_found': len(extracted_data['tables'])
        }
        
    except Exception as e:
        # Create error chart
        fig_mpl, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error processing PDF: {str(e)}\nPlease try with a different PDF file', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title('Processing Error')
        ax.axis('off')
        
        fig_plotly = go.Figure()
        fig_plotly.add_annotation(
            text=f"Error processing PDF: {str(e)}<br>Please try with a different PDF file",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig_plotly.update_layout(title="Processing Error")
        
        return {
            'matplotlib_fig': fig_mpl,
            'plotly_fig': fig_plotly,
            'data_summary': f"Error: {str(e)}",
            'ai_analysis': "Unable to analyze due to processing error.",
            'extracted_numbers': [],
            'tables_found': 0
        }