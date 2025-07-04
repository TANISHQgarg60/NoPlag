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
from typing import List, Tuple, Dict, Optional
import time
import requests
from collections import Counter
import math
from urllib.parse import quote_plus
import json
from bs4 import BeautifulSoup
import urllib.request
from urllib.error import URLError, HTTPError

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        # Fallback to older punkt if punkt_tab fails
        nltk.download('punkt')

class WebSearcher:
    """Web search functionality for plagiarism detection."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def search_bing(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search Bing for potential sources."""
        try:
            # Using Bing search (you could also use other search engines)
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Extract search results
            for result in soup.find_all('li', class_='b_algo')[:num_results]:
                try:
                    title_elem = result.find('h2')
                    url_elem = result.find('a')
                    snippet_elem = result.find('p') or result.find('div', class_='b_caption')
                    
                    if title_elem and url_elem:
                        title = title_elem.get_text(strip=True)
                        url = url_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        # Skip if URL is not valid
                        if not url.startswith('http'):
                            continue
                            
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
                except Exception as e:
                    continue
            
            return results

        except Exception as e:
            print(f"Web search failed: {str(e)}")  # Use print instead of st.warning
            return []
    
    
    def extract_text_from_url(self, url: str) -> str:
        """Extract text content from a URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:10000]  # Limit to first 10k characters
            
        except Exception as e:
            return ""
    
    def search_for_plagiarism(self, sentences: List[str], num_sources: int = 5) -> List[Dict]:
        """Search the web for potential plagiarism sources."""
        web_sources = []
        
        # Select representative sentences for searching
        search_sentences = sentences[:3] if len(sentences) > 3 else sentences
        
        for i, sentence in enumerate(search_sentences):
            # Create search query from sentence (use first 60 characters)
            query = sentence[:60].strip()
            if len(query) < 20:  # Skip very short sentences
                continue
                
            # Search for this sentence
            search_results = self.search_bing(f'"{query}"', num_results=3)
            
            for result in search_results:
                if result['url'] not in [ws['url'] for ws in web_sources]:
                    # Extract content from URL
                    content = self.extract_text_from_url(result['url'])
                    if content:
                        web_sources.append({
                            'title': result['title'],
                            'url': result['url'],
                            'content': content,
                            'snippet': result['snippet'],
                            'search_sentence': sentence
                        })
            
            # Don't exceed the limit
            if len(web_sources) >= num_sources:
                break
                
            # Add delay between searches to be respectful
            time.sleep(1)
        
        return web_sources[:num_sources]

class AIDetector:
    """Simple AI-generated text detector using linguistic patterns."""
    
    def __init__(self):
        self.ai_indicators = {
            'repetitive_phrases': [
                'it is important to note', 'it should be noted', 'furthermore', 
                'moreover', 'in conclusion', 'to summarize', 'in summary',
                'additionally', 'consequently', 'therefore', 'hence'
            ],
            'formal_transitions': [
                'however', 'nevertheless', 'nonetheless', 'subsequently',
                'accordingly', 'thus', 'thereby', 'wherein', 'whereby'
            ],
            'hedging_language': [
                'it appears that', 'it seems that', 'it is likely that',
                'it is possible that', 'it might be', 'it could be'
            ]
        }
    
    def calculate_ai_probability(self, text: str) -> float:
        """Calculate probability that text is AI-generated based on linguistic patterns."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        sentences = sent_tokenize(text)
        
        if not sentences:
            return 0.0
        
        # Calculate various metrics
        scores = []
        
        # 1. Repetitive phrase usage
        phrase_count = sum(text_lower.count(phrase) for phrases in self.ai_indicators.values() 
                          for phrase in phrases)
        phrase_density = phrase_count / len(sentences)
        scores.append(min(phrase_density * 2, 1.0))
        
        # 2. Sentence length uniformity
        sent_lengths = [len(sent.split()) for sent in sentences]
        if len(sent_lengths) > 1:
            length_variance = np.var(sent_lengths)
            avg_length = np.mean(sent_lengths)
            uniformity_score = 1.0 - (length_variance / (avg_length ** 2)) if avg_length > 0 else 0
            scores.append(min(max(uniformity_score, 0), 1.0))
        
        # 3. Vocabulary diversity (inverse of repetition)
        words = re.findall(r'\b\w+\b', text_lower)
        if words:
            unique_words = len(set(words))
            vocab_diversity = unique_words / len(words)
            # AI text often has lower vocabulary diversity
            scores.append(1.0 - vocab_diversity)
        
        # 4. Paragraph structure uniformity
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            para_lengths = [len(para.split()) for para in paragraphs]
            para_uniformity = 1.0 - (np.var(para_lengths) / (np.mean(para_lengths) ** 2)) if np.mean(para_lengths) > 0 else 0
            scores.append(min(max(para_uniformity, 0), 1.0))
        
        return np.mean(scores)

class PlagiarismDetector:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the plagiarism detector with a BERT model."""
        self.model = SentenceTransformer(model_name)
        self.ai_detector = AIDetector()
        self.web_searcher = WebSearcher()
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
    
    def detect_self_plagiarism(self, sentences: List[str], threshold: float = 0.8) -> List[Dict]:
        """Detect self-plagiarism within a single document."""
        if len(sentences) < 2:
            return []
        
        embeddings = self.compute_embeddings(sentences)
        similarity_matrix = cosine_similarity(embeddings, embeddings)
        
        matches = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    matches.append({
                        'sentence_1': sentences[i],
                        'sentence_2': sentences[j],
                        'index_1': i,
                        'index_2': j,
                        'similarity': similarity,
                        'type': 'self_plagiarism'
                    })
        
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)
    
    def search_web_sources(self, sentences: List[str], num_sources: int = 5) -> List[Dict]:
        """Search for potential web sources and analyze them."""
        web_sources = self.web_searcher.search_for_plagiarism(sentences, num_sources)
        
        plagiarism_results = []
        
        for source in web_sources:
            source_sentences = self.extract_sentences(source['content'])
            if source_sentences:
                # Find matches between target sentences and web source
                matches, unique_plagiarized, total_sentences = self.find_similar_sentences(
                    source_sentences, sentences, self.threshold
                )
                
                if matches:
                    plagiarism_results.append({
                        'source': source,
                        'matches': matches,
                        'unique_plagiarized': unique_plagiarized,
                        'total_sentences': total_sentences,
                        'source_sentences': source_sentences
                    })
        
        return plagiarism_results
    
    def find_similar_sentences(self, source_sentences: List[str], 
                             target_sentences: List[str],
                             threshold: float = 0.8) -> Tuple[List[Dict], int, int]:
        """Find similar sentences between source and target texts."""
        if not source_sentences or not target_sentences:
            return [], 0, len(target_sentences) if target_sentences else 0
        
        # Compute embeddings
        source_embeddings = self.compute_embeddings(source_sentences)
        target_embeddings = self.compute_embeddings(target_sentences)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)
        
        matches = []
        matched_target_indices = set()
        
        for i, source_sent in enumerate(source_sentences):
            for j, target_sent in enumerate(target_sentences):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    matches.append({
                        'source_sentence': source_sent,
                        'target_sentence': target_sent,
                        'source_index': i,
                        'target_index': j,
                        'similarity': similarity,
                        'type': 'cross_document'
                    })
                    matched_target_indices.add(j)
        
        # Calculate plagiarism percentage correctly
        unique_plagiarized_sentences = len(matched_target_indices)
        total_target_sentences = len(target_sentences)
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches, unique_plagiarized_sentences, total_target_sentences
    
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

def highlight_matches(text: str, sentences: List[str], match_indices: List[int], 
                     color: str = "#ffeb3b") -> str:
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
            highlighted_sentence = f'<mark style="background-color: {color}; padding: 2px;">{sentence}</mark>'
            highlighted_text = highlighted_text.replace(sentence, highlighted_sentence)
    
    return highlighted_text

def main():
    st.set_page_config(
        page_title="Advanced Plagiarism & AI Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Advanced Plagiarism & AI Detection System")
    st.markdown("Detect plagiarism, self-plagiarism, and AI-generated content using BERT embeddings")
    
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
    
    detection_mode = st.sidebar.selectbox(
        "Detection Mode",
        ["Compare Two Texts", "Web Plagiarism Detection", "Self-Plagiarism Detection", "AI Content Detection", "All Analysis"],
        help="Choose the type of analysis to perform"
    )
    
    enable_web_search = st.sidebar.checkbox(
        "Enable Web Search",
        value=True,
        help="Search the internet for potential plagiarism sources"
    )
    
    num_web_sources = st.sidebar.slider(
        "Number of Web Sources",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of web sources to check for plagiarism"
    ) if enable_web_search else 5
    
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
    
    # Main interface based on detection mode
    if detection_mode == "Compare Two Texts":
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
    
    elif detection_mode == "Web Plagiarism Detection":
        st.header("Text to Check Against Web Sources")
        target_text = st.text_area(
            "Enter the text to check for plagiarism:",
            height=400,
            placeholder="Paste your text here. The system will search the web for potential sources..."
        )
        source_text = ""  # No source text needed for web search
        
        st.info("🌐 This mode will search the internet for potential plagiarism sources automatically.")
    
    elif detection_mode in ["Self-Plagiarism Detection", "AI Content Detection", "All Analysis"]:
        st.header("Text to Analyze")
        target_text = st.text_area(
            "Enter the text to analyze:",
            height=400,
            placeholder="Paste your text here for analysis..."
        )
        source_text = ""  # No source text needed for these modes
    
    # File upload option
    st.header("Or Upload Files")
    uploaded_files = st.file_uploader(
        "Upload text files",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload text files for analysis"
    )
    
    if uploaded_files:
        if detection_mode == "Compare Two Texts":
            if len(uploaded_files) >= 1:
                source_text = uploaded_files[0].read().decode('utf-8')
            if len(uploaded_files) >= 2:
                target_text = uploaded_files[1].read().decode('utf-8')
        else:
            if len(uploaded_files) >= 1:
                target_text = uploaded_files[0].read().decode('utf-8')
    



    # Analysis button
    if st.button("🔍 Start Analysis", type="primary"):
        if not target_text:
            st.error("Please provide text to analyze.")
            return
        
        # Only require source text for direct comparison mode when web search is disabled
        if detection_mode == "Compare Two Texts" and not source_text:
            st.error("Please provide both source and target texts for comparison.")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Extract sentences
            status_text.text("Extracting sentences...")
            progress_bar.progress(20)
            
            target_sentences = detector.extract_sentences(target_text)
            
            if not target_sentences:
                st.error("Could not extract sentences from the provided text.")
                return
            
            results = {}
            
            

            
            # Web plagiarism detection
            if (detection_mode in ["Web Plagiarism Detection", "All Analysis"] and enable_web_search):
                
                status_text.text("🌐 Searching web for potential sources...")
                progress_bar.progress(30)
                
                web_results = detector.search_web_sources(target_sentences, num_web_sources)
                # Debug output
                st.write(f"Debug: Found {len(web_results)} web sources")
                for i, result in enumerate(web_results):
                    st.write(f"Source {i+1}: {len(result.get('matches', []))} matches")
                results['web_plagiarism'] = web_results
                
                if web_results:
                    status_text.text("📝 Analyzing web sources...")
                    progress_bar.progress(50)



            
            
            # AI Detection
            if detection_mode in ["AI Content Detection", "All Analysis"]:
                status_text.text("🤖 Analyzing AI-generated content...")
                progress_bar.progress(60)
                
                ai_probability = detector.ai_detector.calculate_ai_probability(target_text)
                results['ai_detection'] = {
                    'probability': ai_probability,
                    'confidence': 'High' if ai_probability > 0.7 else 'Medium' if ai_probability > 0.4 else 'Low'
                }
            
            # Self-plagiarism detection
            if detection_mode in ["Self-Plagiarism Detection", "All Analysis"]:
                status_text.text("🔄 Detecting self-plagiarism...")
                progress_bar.progress(70)
                
                self_matches = detector.detect_self_plagiarism(target_sentences, threshold)
                results['self_plagiarism'] = self_matches
            
            # Cross-document plagiarism detection
            if detection_mode in ["Compare Two Texts", "All Analysis"] and source_text:
                status_text.text("📋 Comparing with source text...")
                progress_bar.progress(80)
                
                source_sentences = detector.extract_sentences(source_text)
                if source_sentences:
                    try:
                        result = detector.find_similar_sentences(
                            source_sentences, target_sentences, threshold
                        )
                        # Handle different return formats
                        if isinstance(result, tuple) and len(result) == 3:
                            matches, unique_plagiarized, total_sentences = result
                        else:
                            # Fallback if method returns old format
                            matches = result if isinstance(result, list) else []
                            unique_plagiarized = len(set([m['target_index'] for m in matches]))
                            total_sentences = len(target_sentences)
                        
                        results['cross_document'] = {
                            'matches': matches,
                            'unique_plagiarized': unique_plagiarized,
                            'total_sentences': total_sentences
                        }
                        
                        # Similarity distribution
                        similarities = detector.analyze_similarity_distribution(
                            source_sentences, target_sentences
                        )
                        results['similarities'] = similarities
                    except Exception as e:
                        st.error(f"Error in cross-document analysis: {str(e)}")
                        results['cross_document'] = {
                            'matches': [],
                            'unique_plagiarized': 0,
                            'total_sentences': len(target_sentences)
                        }
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Summary metrics
            cols = st.columns(5)
            
            with cols[0]:
                if 'cross_document' in results:
                    plagiarism_percentage = (results['cross_document']['unique_plagiarized'] / 
                                           results['cross_document']['total_sentences']) * 100
                    st.metric("Cross-Doc Plagiarism %", f"{plagiarism_percentage:.1f}%")

                elif 'web_plagiarism' in results and results['web_plagiarism']:
                    # Calculate overall web plagiarism percentage
                    all_matches = []
                    for web_result in results['web_plagiarism']:
                        if 'matches' in web_result and web_result['matches']:
                            all_matches.extend([m['target_index'] for m in web_result['matches']])
                    unique_web_matches = len(set(all_matches)) if all_matches else 0
                    web_plag_percentage = (unique_web_matches / len(target_sentences)) * 100 if len(target_sentences) > 0 else 0
                    st.metric("Web Plagiarism %", f"{web_plag_percentage:.1f}%")
                elif 'self_plagiarism' in results:
                    self_plag_sentences = len(set([m['index_1'] for m in results['self_plagiarism']] + 
                                                [m['index_2'] for m in results['self_plagiarism']]))
                    self_plag_percentage = (self_plag_sentences / len(target_sentences)) * 100
                    st.metric("Self-Plagiarism %", f"{self_plag_percentage:.1f}%")
            
            with cols[1]:
                if 'ai_detection' in results:
                    ai_prob = results['ai_detection']['probability']
                    st.metric("AI Probability", f"{ai_prob:.1%}")
            
            with cols[2]:
                if 'cross_document' in results:
                    st.metric("Cross-Doc Matches", len(results['cross_document']['matches']))
                elif 'web_plagiarism' in results:
                    total_web_matches = sum(len(wr['matches']) for wr in results['web_plagiarism'])
                    st.metric("Web Matches", total_web_matches)
                if 'self_plagiarism' in results:
                    st.metric("Self-Matches", len(results['self_plagiarism']))
            
            with cols[3]:
                if 'web_plagiarism' in results:
                    st.metric("Web Sources Found", len(results['web_plagiarism']))
                st.metric("Total Sentences", len(target_sentences))
            
            with cols[4]:
                if 'web_plagiarism' in results and results['web_plagiarism']:
                    # Show highest similarity from web sources
                    max_similarity = max(
                        max([m['similarity'] for m in wr['matches']], default=0)
                        for wr in results['web_plagiarism']
                    )
                    st.metric("Max Web Similarity", f"{max_similarity:.3f}")
                elif 'cross_document' in results and results['cross_document']['matches']:
                    max_similarity = max(m['similarity'] for m in results['cross_document']['matches'])
                    st.metric("Max Similarity", f"{max_similarity:.3f}")
            
            # AI Detection Results
            if 'ai_detection' in results:
                st.subheader("🤖 AI Content Detection")
                ai_data = results['ai_detection']
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Create AI probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = ai_data['probability'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "AI Generation Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Analysis:**")
                    if ai_data['probability'] > 0.7:
                        st.error(f"🚨 High probability ({ai_data['probability']:.1%}) of AI-generated content")
                    elif ai_data['probability'] > 0.4:
                        st.warning(f"⚠️ Medium probability ({ai_data['probability']:.1%}) of AI-generated content")
                    else:
                        st.success(f"✅ Low probability ({ai_data['probability']:.1%}) of AI-generated content")
            
            # Web plagiarism results
            if 'web_plagiarism' in results and results['web_plagiarism']:
                st.subheader("🌐 Web Plagiarism Detection Results")
                
                # Overview of web sources
                st.write("**Sources Found:**")
                for i, web_result in enumerate(results['web_plagiarism'], 1):
                    source = web_result['source']
                    matches_count = len(web_result['matches'])
                    if matches_count > 0:
                        similarity_scores = [m['similarity'] for m in web_result['matches']]
                        avg_similarity = np.mean(similarity_scores)
                        max_similarity = max(similarity_scores)
                        
                        with st.expander(f"🔍 Source {i}: {source['title']} ({matches_count} matches)"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**URL:** {source['url']}")
                                st.markdown(f"**Snippet:** {source['snippet']}")
                                st.markdown(f"**Matches:** {matches_count}")
                                st.markdown(f"**Avg Similarity:** {avg_similarity:.3f}")
                                st.markdown(f"**Max Similarity:** {max_similarity:.3f}")
                            
                            with col2:
                                st.markdown("**Sample Matches:**")
                                for j, match in enumerate(web_result['matches'][:3]):
                                    st.markdown(f"**Match {j+1}** (sim: {match['similarity']:.3f})")
                                    st.markdown(f"*Your text:* {match['target_sentence'][:100]}...")
                                    st.markdown(f"*Source:* {match['source_sentence'][:100]}...")
                                    st.markdown("---")
                
                # Detailed matches table
                # Detailed matches table
                st.subheader("📋 Detailed Web Matches")
                
                all_web_matches = []
                for web_result in results['web_plagiarism']:
                    for match in web_result['matches']:
                        all_web_matches.append({
                            'Source': web_result['source']['title'],
                            'URL': web_result['source']['url'],
                            'Target Sentence': match['target_sentence'][:100] + "..." if len(match['target_sentence']) > 100 else match['target_sentence'],
                            'Source Sentence': match['source_sentence'][:100] + "..." if len(match['source_sentence']) > 100 else match['source_sentence'],
                            'Similarity': f"{match['similarity']:.3f}"
                        })
                
                if all_web_matches:
                    matches_df = pd.DataFrame(all_web_matches)
                    st.dataframe(matches_df, use_container_width=True)
                    
                    # Download matches as CSV
                    csv = matches_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Web Matches as CSV",
                        data=csv,
                        file_name="web_plagiarism_matches.csv",
                        mime="text/csv"
                    )
            
            # Cross-document plagiarism results
            if 'cross_document' in results:
                st.subheader("📋 Cross-Document Plagiarism Results")
                
                cross_data = results['cross_document']
                if cross_data['matches']:
                    # Create matches dataframe
                    matches_data = []
                    for match in cross_data['matches']:
                        matches_data.append({
                            'Source Sentence': match['source_sentence'][:100] + "..." if len(match['source_sentence']) > 100 else match['source_sentence'],
                            'Target Sentence': match['target_sentence'][:100] + "..." if len(match['target_sentence']) > 100 else match['target_sentence'],
                            'Similarity': f"{match['similarity']:.3f}",
                            'Source Index': match['source_index'],
                            'Target Index': match['target_index']
                        })
                    
                    matches_df = pd.DataFrame(matches_data)
                    st.dataframe(matches_df, use_container_width=True)
                    
                    # Download matches as CSV
                    csv = matches_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cross-Document Matches as CSV",
                        data=csv,
                        file_name="cross_document_matches.csv",
                        mime="text/csv"
                    )
                    
                    # Show similarity distribution
                    if show_distribution and 'similarities' in results:
                        st.subheader("📊 Similarity Score Distribution")
                        fig = create_similarity_histogram(results['similarities'], threshold)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show highlighted text
                    if show_highlights:
                        st.subheader("🖍️ Highlighted Matches in Target Text")
                        match_indices = [m['target_index'] for m in cross_data['matches']]
                        highlighted_target = highlight_matches(target_text, target_sentences, match_indices)
                        st.markdown(highlighted_target, unsafe_allow_html=True)
                        
                        st.subheader("🖍️ Highlighted Matches in Source Text")
                        match_indices = [m['source_index'] for m in cross_data['matches']]
                        source_sentences = detector.extract_sentences(source_text)
                        highlighted_source = highlight_matches(source_text, source_sentences, match_indices, "#ff9800")
                        st.markdown(highlighted_source, unsafe_allow_html=True)
                else:
                    st.success("✅ No cross-document plagiarism detected!")
            
            # Self-plagiarism results
            if 'self_plagiarism' in results:
                st.subheader("🔄 Self-Plagiarism Detection Results")
                
                self_matches = results['self_plagiarism']
                if self_matches:
                    # Create self-plagiarism dataframe
                    self_data = []
                    for match in self_matches:
                        self_data.append({
                            'First Sentence': match['sentence_1'][:100] + "..." if len(match['sentence_1']) > 100 else match['sentence_1'],
                            'Second Sentence': match['sentence_2'][:100] + "..." if len(match['sentence_2']) > 100 else match['sentence_2'],
                            'Similarity': f"{match['similarity']:.3f}",
                            'First Index': match['index_1'],
                            'Second Index': match['index_2']
                        })
                    
                    self_df = pd.DataFrame(self_data)
                    st.dataframe(self_df, use_container_width=True)
                    
                    # Download self-plagiarism matches as CSV
                    csv = self_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Self-Plagiarism Matches as CSV",
                        data=csv,
                        file_name="self_plagiarism_matches.csv",
                        mime="text/csv"
                    )
                    
                    # Show highlighted text for self-plagiarism
                    if show_highlights:
                        st.subheader("🖍️ Highlighted Self-Plagiarism in Text")
                        all_match_indices = []
                        for match in self_matches:
                            all_match_indices.extend([match['index_1'], match['index_2']])
                        highlighted_text = highlight_matches(target_text, target_sentences, all_match_indices, "#e91e63")
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                else:
                    st.success("✅ No self-plagiarism detected!")
            
            # Text Statistics
            st.subheader("📈 Text Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Characters", len(target_text))
            with col2:
                word_count = len(target_text.split())
                st.metric("Total Words", word_count)
            with col3:
                st.metric("Total Sentences", len(target_sentences))
            with col4:
                avg_sentence_length = word_count / len(target_sentences) if target_sentences else 0
                st.metric("Avg Sentence Length", f"{avg_sentence_length:.1f}")
            
            # Additional insights
            st.subheader("💡 Analysis Insights")
            
            insights = []
            
            # AI Detection insights
            if 'ai_detection' in results:
                ai_prob = results['ai_detection']['probability']
                if ai_prob > 0.7:
                    insights.append("🤖 **High AI probability detected**: The text shows strong patterns typical of AI-generated content, including formal language patterns and uniform sentence structures.")
                elif ai_prob > 0.4:
                    insights.append("🤖 **Moderate AI probability**: Some patterns suggest possible AI involvement, but human authorship is still likely.")
                else:
                    insights.append("✅ **Low AI probability**: The text appears to be human-written with natural language patterns.")
            
            # Web plagiarism insights
            if 'web_plagiarism' in results and results['web_plagiarism']:
                total_web_matches = sum(len(wr['matches']) for wr in results['web_plagiarism'])
                unique_web_matches = len(set(sum([[m['target_index'] for m in wr['matches']] for wr in results['web_plagiarism']], [])))
                web_plag_percentage = (unique_web_matches / len(target_sentences)) * 100
                
                if web_plag_percentage > 20:
                    insights.append(f"🚨 **High web plagiarism detected**: {web_plag_percentage:.1f}% of sentences show similarity to web sources. Consider reviewing and properly citing sources.")
                elif web_plag_percentage > 10:
                    insights.append(f"⚠️ **Moderate web plagiarism detected**: {web_plag_percentage:.1f}% of sentences show similarity to web sources. Some citations may be needed.")
                else:
                    insights.append(f"✅ **Low web plagiarism**: Only {web_plag_percentage:.1f}% similarity to web sources detected.")
            
            # Cross-document insights
            if 'cross_document' in results:
                cross_data = results['cross_document']
                plag_percentage = (cross_data['unique_plagiarized'] / cross_data['total_sentences']) * 100
                
                if plag_percentage > 30:
                    insights.append(f"🚨 **High cross-document plagiarism**: {plag_percentage:.1f}% of sentences match the source document.")
                elif plag_percentage > 15:
                    insights.append(f"⚠️ **Moderate cross-document plagiarism**: {plag_percentage:.1f}% of sentences show similarity to the source.")
                else:
                    insights.append(f"✅ **Low cross-document plagiarism**: {plag_percentage:.1f}% similarity detected.")
            
            # Self-plagiarism insights
            if 'self_plagiarism' in results:
                self_matches = results['self_plagiarism']
                if len(self_matches) > 5:
                    insights.append(f"🔄 **Significant self-plagiarism detected**: {len(self_matches)} instances of repeated content found within the document.")
                elif len(self_matches) > 0:
                    insights.append(f"🔄 **Minor self-plagiarism detected**: {len(self_matches)} instances of repeated content found.")
                else:
                    insights.append("✅ **No self-plagiarism detected**: Content appears to be original throughout.")
            
            # Display insights
            for insight in insights:
                st.markdown(insight)
            
            # Recommendations
            st.subheader("📋 Recommendations")
            recommendations = []
            
            # Based on AI detection
            if 'ai_detection' in results and results['ai_detection']['probability'] > 0.5:
                recommendations.append("Consider adding more personal voice and varied sentence structures to make the text appear more naturally human-written.")
            
            # Based on plagiarism
            if any([
                'web_plagiarism' in results and results['web_plagiarism'],
                'cross_document' in results and results['cross_document']['matches'],
                'self_plagiarism' in results and results['self_plagiarism']
            ]):
                recommendations.append("Review flagged sections and ensure proper citations are included for any borrowed content.")
                recommendations.append("Consider paraphrasing similar content to increase originality.")
                recommendations.append("Use quotation marks for direct quotes and provide appropriate attributions.")
            
            if not recommendations:
                recommendations.append("The text appears to be original. Continue maintaining good academic integrity practices.")
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.error("Please check your input and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🔍 <strong>Advanced Plagiarism & AI Detection System</strong></p>
        <p>Uses BERT embeddings for semantic similarity analysis | Web search for plagiarism detection | AI pattern recognition</p>
        <p><small>⚠️ This tool is for educational and research purposes. Always verify results and consult institutional guidelines.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
