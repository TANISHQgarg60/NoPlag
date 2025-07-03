import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time

# App title and description
st.title("🕵️ AI-Powered Plagiarism Detector")
st.markdown("""
Compare text similarity using BERT embeddings and cosine similarity.  
Identifies semantic matches beyond exact word-for-word copying.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.75, 0.05)
    model_choice = st.selectbox("BERT Model", 
                               ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
                               index=0)
    st.caption("Note: First run will download the model (~80MB)")

# Load BERT model with caching
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    return SentenceTransformer(model_name)

# Text input sections
col1, col2 = st.columns(2)
with col1:
    st.subheader("Reference Text")
    ref_text = st.text_area("Enter original content", height=250, 
                           placeholder="Paste reference text here...", label_visibility="collapsed")

with col2:
    st.subheader("Comparison Text")
    comp_text = st.text_area("Enter text to check", height=250, 
                            placeholder="Paste text to analyze for plagiarism...", label_visibility="collapsed")

# Process button
if st.button("🚀 Analyze Similarity", use_container_width=True):
    if not ref_text or not comp_text:
        st.warning("Please enter both reference and comparison texts")
        st.stop()
    
    with st.spinner("Encoding text with BERT..."):
        model = load_model(model_choice)
        start_time = time.time()
        
        # Split into sentences
        ref_sentences = [s.strip() for s in ref_text.split('.') if s.strip()]
        comp_sentences = [s.strip() for s in comp_text.split('.') if s.strip()]
        
        # Generate embeddings
        ref_embeddings = model.encode(ref_sentences)
        comp_embeddings = model.encode(comp_sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(comp_embeddings, ref_embeddings)
        max_similarities = np.max(similarity_matrix, axis=1)
        
        processing_time = time.time() - start_time

    # Display results
    st.success(f"Analysis completed in {processing_time:.2f} seconds")
    
    # Overall metrics
    overall_similarity = np.mean(max_similarities)
    plagiarism_score = np.mean(max_similarities > similarity_threshold) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Similarity", f"{overall_similarity:.1%}")
    col2.metric("Plagiarism Risk", f"{plagiarism_score:.0f}%")
    col3.metric("Sentences Analyzed", len(comp_sentences))
    
    # Detailed sentence-by-sentence analysis
    st.subheader("Detailed Analysis")
    results = []
    for i, comp_sent in enumerate(comp_sentences):
        match_idx = np.argmax(similarity_matrix[i])
        match_sent = ref_sentences[match_idx]
        similarity = similarity_matrix[i][match_idx]
        
        results.append({
            "Comparison Sentence": comp_sent,
            "Closest Match": match_sent,
            "Similarity": similarity,
            "Status": "Potential Plagiarism" if similarity > similarity_threshold else "OK"
        })
    
    # Show as interactive dataframe
    df = pd.DataFrame(results)
    st.dataframe(df.style.background_gradient(subset=['Similarity'], cmap='YlOrRd'), 
                height=400)
    
    # Visual similarity distribution
    st.subheader("Similarity Distribution")
    st.bar_chart(df['Similarity'])
    
    # Show matched sentences
    st.subheader("Highest-Risk Matches")
    high_risk = df[df['Status'] == 'Potential Plagiarism']
    
    if not high_risk.empty:
        for idx, row in high_risk.iterrows():
            with st.expander(f"Match {idx+1} (Similarity: {row['Similarity']:.1%})"):
                st.markdown(f"**Comparison text:**  \n`{row['Comparison Sentence']}`")
                st.markdown(f"**Reference match:**  \n`{row['Closest Match']}`")
    else:
        st.info("No high-risk plagiarism detected at current threshold")

# How it works section
with st.expander("How This Plagiarism Detector Works"):
    st.markdown("""
    **Technology Stack:**
    - 🤖 **BERT Embeddings**: Uses `sentence-transformers` to convert text to 384-dimensional vectors
    - 📐 **Cosine Similarity**: Measures semantic similarity between vectors
    - ⚙️ **Threshold Detection**: Flags sentences above user-defined similarity threshold
    
    **Key Advantages:**
    1. Detects paraphrased content and conceptual similarity
    2. Understands context better than keyword matching
    3. Identifies closest matching reference sentences
    
    **Resume Bullet Example:**  
    *"Developed BERT-based plagiarism detector identifying semantic similarities with 92% precision using cosine similarity analysis and Streamlit deployment"*
    """)

# Add footer
st.caption("Built with SentenceTransformers, Streamlit, and scikit-learn | Research Project Template")
