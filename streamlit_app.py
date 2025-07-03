import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import plotly.express as px
import plotly.graph_objects as go
import re
import io
from typing import List, Tuple, Dict
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        # Fallback to older punkt if punkt_tab fails
        nltk.download('punkt')

class PlagiarismDetector:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the plagiarism detector with a BERT model."""
        self.model = SentenceTransformer(model_name)
        self.threshold = 0.8
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep punctuation for sentence segmentation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        preprocessed = self.preprocess_text(text)
        sentences = sent_tokenize(preprocessed)
        # Filter out very short sentences (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute BERT embeddings for sentences."""
        return self.model.encode(sentences)
    
    def find_similar_sentences(self, source_sentences: List[str], 
                             target_sentences: List[str],
                             threshold: float = 0.8) -> List[Dict]:
        """Find similar sentences between source and target texts."""
        if not source_sentences or not target_sentences:
            return []
        
        # Compute embeddings
        source_embeddings = self.compute_embeddings(source_sentences)
        target_embeddings = self.compute_embeddings(target_sentences)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)
        
        matches = []
        for i, source_sent in enumerate(source_sentences):
            for j, target_sent in enumerate(target_sentences):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    matches.append({
                        'source_sentence': source_sent,
                        'target_sentence': target_sent,
                        'source_index': i,
                        'target_index': j,
                        'similarity': similarity
                    })
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
    
    def analyze_similarity_distribution(self, source_sentences: List[str], 
                                     target_sentences: List[str]) -> np.ndarray:
        """Analyze the distribution of similarity scores."""
        if not source_sentences or not target_sentences:
            return np.array([])
        
        source_embeddings = self.compute_embeddings(source_sentences)
        target_embeddings = self.compute_embeddings(target_sentences)
        similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)
        
        return similarity_matrix.flatten()

def create_similarity_histogram(similarities: np.ndarray, threshold: float):
    """Create a histogram of similarity scores."""
    if len(similarities) == 0:
        return go.Figure()
    
    fig = px.histogram(
        x=similarities,
        nbins=50,
        title="Distribution of Sentence Similarity Scores",
        labels={'x': 'Cosine Similarity', 'y': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold:.2f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        xaxis_title="Cosine Similarity Score",
        yaxis_title="Frequency",
        showlegend=False
    )
    
    return fig

def highlight_matches(text: str, sentences: List[str], match_indices: List[int]) -> str:
    """Highlight matching sentences in text."""
    if not match_indices:
        return text
    
    # Create a copy of the text for highlighting
    highlighted_text = text
    
    # Sort indices in reverse order to avoid index shifting issues
    sorted_indices = sorted(set(match_indices), reverse=True)
    
    for idx in sorted_indices:
        if idx < len(sentences):
            sentence = sentences[idx]
            # Use HTML highlighting
            highlighted_sentence = f'<mark style="background-color: #ffeb3b; padding: 2px;">{sentence}</mark>'
            highlighted_text = highlighted_text.replace(sentence, highlighted_sentence)
    
    return highlighted_text

def main():
    st.set_page_config(
        page_title="BERT Plagiarism Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 BERT-Based Plagiarism Detector")
    st.markdown("Detect potential plagiarism using BERT sentence embeddings and cosine similarity")
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        return PlagiarismDetector()
    
    detector = load_detector()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Sentences with similarity above this threshold will be flagged"
    )
    
    show_distribution = st.sidebar.checkbox(
        "Show Similarity Distribution",
        value=True,
        help="Display histogram of all similarity scores"
    )
    
    show_highlights = st.sidebar.checkbox(
        "Highlight Matches",
        value=True,
        help="Highlight matching sentences in the text"
    )
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Source Text")
        source_text = st.text_area(
            "Enter the source text to check against:",
            height=300,
            placeholder="Paste your source text here..."
        )
    
    with col2:
        st.header("Target Text")
        target_text = st.text_area(
            "Enter the text to check for plagiarism:",
            height=300,
            placeholder="Paste your target text here..."
        )
    
    # File upload option
    st.header("Or Upload Files")
    uploaded_files = st.file_uploader(
        "Upload text files",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload up to 2 text files (first will be source, second will be target)"
    )
    
    if uploaded_files:
        if len(uploaded_files) >= 1:
            source_text = uploaded_files[0].read().decode('utf-8')
        if len(uploaded_files) >= 2:
            target_text = uploaded_files[1].read().decode('utf-8')
    
    # Analysis button
    if st.button("🔍 Analyze Plagiarism", type="primary"):
        if not source_text or not target_text:
            st.error("Please provide both source and target texts.")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Extract sentences
            status_text.text("Extracting sentences...")
            progress_bar.progress(20)
            
            source_sentences = detector.extract_sentences(source_text)
            target_sentences = detector.extract_sentences(target_text)
            
            if not source_sentences or not target_sentences:
                st.error("Could not extract sentences from the provided texts.")
                return
            
            # Find matches
            status_text.text("Computing embeddings and finding matches...")
            progress_bar.progress(60)
            
            matches = detector.find_similar_sentences(
                source_sentences, target_sentences, threshold
            )
            
            # Analyze distribution
            status_text.text("Analyzing similarity distribution...")
            progress_bar.progress(80)
            
            similarities = detector.analyze_similarity_distribution(
                source_sentences, target_sentences
            )
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Matches", len(matches))
            
            with col2:
                plagiarism_percentage = (len(matches) / len(target_sentences)) * 100 if target_sentences else 0
                st.metric("Plagiarism %", f"{plagiarism_percentage:.1f}%")
            
            with col3:
                avg_similarity = np.mean([m['similarity'] for m in matches]) if matches else 0
                st.metric("Avg Similarity", f"{avg_similarity:.3f}")
            
            with col4:
                max_similarity = max([m['similarity'] for m in matches]) if matches else 0
                st.metric("Max Similarity", f"{max_similarity:.3f}")
            
            # Similarity distribution
            if show_distribution and len(similarities) > 0:
                st.subheader("Similarity Score Distribution")
                fig = create_similarity_histogram(similarities, threshold)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed matches
            if matches:
                st.subheader("🎯 Detected Matches")
                
                # Create matches dataframe
                matches_df = pd.DataFrame([
                    {
                        'Similarity': f"{m['similarity']:.3f}",
                        'Source Sentence': m['source_sentence'][:100] + "..." if len(m['source_sentence']) > 100 else m['source_sentence'],
                        'Target Sentence': m['target_sentence'][:100] + "..." if len(m['target_sentence']) > 100 else m['target_sentence']
                    }
                    for m in matches[:20]  # Show top 20 matches
                ])
                
                st.dataframe(matches_df, use_container_width=True)
                
                if len(matches) > 20:
                    st.info(f"Showing top 20 matches out of {len(matches)} total matches.")
                
                # Highlighted text
                if show_highlights:
                    st.subheader("📝 Highlighted Texts")
                    
                    source_match_indices = [m['source_index'] for m in matches]
                    target_match_indices = [m['target_index'] for m in matches]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Source Text (with matches highlighted):**")
                        highlighted_source = highlight_matches(
                            source_text, source_sentences, source_match_indices
                        )
                        st.markdown(highlighted_source, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Target Text (with matches highlighted):**")
                        highlighted_target = highlight_matches(
                            target_text, target_sentences, target_match_indices
                        )
                        st.markdown(highlighted_target, unsafe_allow_html=True)
                
                # Export results
                st.subheader("💾 Export Results")
                
                # Create detailed report
                report_data = []
                for match in matches:
                    report_data.append({
                        'Similarity_Score': match['similarity'],
                        'Source_Sentence': match['source_sentence'],
                        'Target_Sentence': match['target_sentence'],
                        'Source_Index': match['source_index'],
                        'Target_Index': match['target_index']
                    })
                
                report_df = pd.DataFrame(report_data)
                
                # Convert to CSV
                csv_buffer = io.StringIO()
                report_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Detailed Report (CSV)",
                    data=csv_data,
                    file_name="plagiarism_report.csv",
                    mime="text/csv"
                )
            
            else:
                st.success("🎉 No plagiarism detected above the specified threshold!")
                st.info("Try lowering the similarity threshold to see more potential matches.")
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.error("Please check your input texts and try again.")

if __name__ == "__main__":
    main()
