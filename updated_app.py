# Add this import at the top with other imports
from visualization import generate_visual_insights

# Update the task selection dropdown to include the new feature
if "documents" in st.session_state:
    st.subheader("🎓 Master Agent: What would you like me to do?")
    task = st.selectbox("Choose a task:", [
        "Summarize document",
        "Identify research gaps",
        "Suggest research ideas",
        "Simulate a debate",
        "Generate citation",
        "Generate visual insights",  # NEW FEATURE ADDED
        "Chat with paper"
    ])

    # Add the new visualization handling in the main button logic
    elif st.button("🚀 Run Agent"):
        with st.spinner("Running agents..."):
            docs = st.session_state.documents[:10]
            output = ""

            if task == "Summarize document":
                output = summarize_document(llm, docs)

            elif task == "Identify research gaps":
                output = identify_research_gaps(llm, docs)

            elif task == "Suggest research ideas":
                output = suggest_research_ideas(llm, docs)

            elif task == "Simulate a debate":
                output = simulate_debate(llm, docs)

            elif task == "Generate citation":
                output = generate_citation(llm, docs)

            elif task == "Generate visual insights":  # NEW FEATURE HANDLER
                with st.spinner("Extracting data and generating visualizations..."):
                    # Get the original uploaded files from session state
                    if "uploaded_files" in st.session_state:
                        insights = generate_visual_insights(llm, st.session_state.uploaded_files)
                        st.session_state["visual_insights"] = insights
                        output = insights['ai_analysis']
                    else:
                        output = "Error: Please re-upload your PDF files to use this feature."

            if output:
                st.session_state["last_agent_output"] = output

# Add this section after the file uploader to store uploaded files
if uploaded_files and st.button("📚 Process Documents"):
    with st.spinner("Processing documents and generating vector store..."):
        documents = process_pdfs(uploaded_files)
        st.session_state.documents = documents
        st.session_state.vectorstore = create_vector_store(documents, embedding)
        st.session_state.uploaded_files = uploaded_files  # Store uploaded files for visualization
    st.success("✅ Document vector store created!")

# Add this section after the translation section to display visualizations
# Display Visual Insights if available
if "visual_insights" in st.session_state:
    insights = st.session_state["visual_insights"]
    
    st.markdown("### 📊 Visual Insights")
    
    # Display data summary
    st.markdown("#### Data Extraction Summary")
    st.info(f"📋 {insights['data_summary']}")
    st.info(f"📊 Tables found: {insights['tables_found']}")
    
    if insights['extracted_numbers']:
        st.info(f"🔢 Sample extracted numbers: {', '.join(insights['extracted_numbers'][:10])}")
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Interactive Chart (Plotly)")
        st.plotly_chart(insights['plotly_fig'], use_container_width=True)
    
    with col2:
        st.markdown("#### Static Chart (Matplotlib)")
        st.pyplot(insights['matplotlib_fig'])
    
    # Display AI analysis
    st.markdown("#### 🤖 AI Analysis of Visual Data")
    st.write(insights['ai_analysis'])
    
    # Clear button
    if st.button("🗑️ Clear Visual Insights"):
        del st.session_state["visual_insights"]
        st.rerun()