import streamlit as st
import os
import tempfile

# Force PyTorch and disable TensorFlow completely
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["NO_TF"] = "1"

# Import after setting environment
from utils.audio_processor import AudioProcessor
from utils.transcription import TranscriptGenerator
from utils.summarization import SummaryGenerator
from utils.visualization import TextVisualizer  # NEW IMPORT
from datetime import datetime

# Page configuration - FULL SCREEN
st.set_page_config(
    page_title="Intelligent Meeting Notes Generator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed
)

# Custom CSS for full-screen layout
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        max-width: 100%;
    }
    .summary-card {
        background: white;
        border-radius: 10px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .point-item {
        padding: 8px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .point-item:last-child {
        border-bottom: none;
    }
    .action-item {
        background: #fff3cd;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #ffc107;
    }
    .decision-item {
        background: #d1ecf1;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #17a2b8;
    }
    .key-point-item {
        background: #f8f9fa;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #28a745;
    }
    .viz-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header - Full Width
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 3rem; color: #333; margin-bottom: 0.5rem;">🎙️ Intelligent Meeting Notes Generator</h1>
            <p style="font-size: 1.2rem; color: #666;">Transform your meeting audio into structured, actionable notes</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h3>🤖 AI Meeting Assistant</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.title("Navigation")
        app_mode = st.selectbox("Choose Mode", ["Upload Audio", "About Project"])
        
        if app_mode == "Upload Audio":
            st.markdown("---")
            st.subheader("⚙️ Model Settings")
            
            # Model selection
            transcription_model = st.selectbox(
                "Transcription Model",
                ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"]
            )
            
            summarization_model = st.selectbox(
                "Summarization Model",
                ["lsa", "textrank"]
            )
            
            return transcription_model, summarization_model, app_mode
        else:
            return None, None, app_mode

def upload_audio_interface(transcription_model, summarization_model):
    """Main interface for audio upload and processing - FULL SCREEN"""
    
    # File upload section - Full width
    st.markdown("""
    <div class="summary-card">
        <h2 style="color: #2c3e50; margin-bottom: 10px;">📁 Upload Meeting Audio</h2>
        <p style="color: #7f8c8d;">Upload your meeting recording to generate intelligent notes</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Supported formats: WAV, MP3, M4A, FLAC"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                    border-radius: 10px; padding: 20px; margin: 15px 0; color: white;">
            <h4 style="margin-bottom: 15px;">📄 File Details</h4>
        """, unsafe_allow_html=True)
        
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process button - Centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Meeting Notes", type="primary", use_container_width=True):
                return uploaded_file, transcription_model, summarization_model
    
    return None, None, None

def display_results_fullscreen(transcript, summary_result):
    """Display results in full-screen layout"""
    
    # Main Results Header - BLUE BOX
    st.markdown("""
    <div class="section-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <h2 style="margin: 0; text-align: center;">🎯 Intelligent Meeting Notes Generated</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create two main columns for transcript and summary
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Transcript Section - Full height with BLUE HEADER
        st.markdown("""
        <div class="section-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3 style="margin: 0;">📝 Full Meeting Transcript</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.text_area(
            "Transcript", 
            transcript, 
            height=400,
            label_visibility="collapsed"
        )
    
    with col2:
        # Summary Section - Full height with BLUE HEADER
        st.markdown("""
        <div class="section-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3 style="margin: 0;">📋 Executive Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Overall Summary
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        if summary_result["overall_summary"]:
            summary_points = clean_text_for_display(summary_result["overall_summary"])
            for point in summary_points:
                if point.strip():
                    st.markdown(f"<div class='point-item'>• {point.strip()}</div>", unsafe_allow_html=True)
        else:
            st.info("No overall summary generated.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second row - Action Items and Decisions
    col3, col4 = st.columns([1, 1])
    
    with col3:
        # Action Items with BLUE HEADER
        st.markdown("""
        <div class="section-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3 style="margin: 0;">✅ Action Items</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        if summary_result["action_items"] and summary_result["action_items"] != "No specific action items identified.":
            action_points = clean_text_for_display(summary_result["action_items"])
            for action in action_points:
                if action.strip():
                    st.markdown(f"<div class='action-item'>☐ {action.strip()}</div>", unsafe_allow_html=True)
        else:
            st.info("No specific action items identified.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        # Decisions with BLUE HEADER
        st.markdown("""
        <div class="section-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3 style="margin: 0;">🎯 Decisions Made</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        if summary_result["decisions"] and summary_result["decisions"] != "No specific decisions identified.":
            decision_points = clean_text_for_display(summary_result["decisions"])
            for decision in decision_points:
                if decision.strip():
                    st.markdown(f"<div class='decision-item'>✓ {decision.strip()}</div>", unsafe_allow_html=True)
        else:
            st.info("No specific decisions identified.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Third row - Key Points (Full width) with BLUE HEADER
    st.markdown("""
    <div class="section-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <h3 style="margin: 0;">📊 Key Discussion Points</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    if summary_result["key_points"] and summary_result["key_points"] != "No key points identified.":
        key_points_list = clean_text_for_display(summary_result["key_points"])
        for i, point in enumerate(key_points_list, 1):
            if point.strip():
                st.markdown(f"<div class='key-point-item'>{i}. {point.strip()}</div>", unsafe_allow_html=True)
    else:
        st.info("No key points identified.")
    st.markdown('</div>', unsafe_allow_html=True)

def display_visualizations(transcript):
    """Display word cloud and frequency analysis"""
    
    st.markdown("---")
    st.markdown("""
    <div class="section-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <h2 style="margin: 0; text-align: center;">📊 Text Analysis & Insights</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize visualizer
    visualizer = TextVisualizer()
    
    # Generate visualizations
    with st.spinner("🔄 Generating text insights and visualizations..."):
        viz_data = visualizer.create_visualization_section(transcript)
    
    if viz_data:
        # Display in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ☁️ Word Cloud Analysis")
            if viz_data['wordcloud']:
                st.image(viz_data['wordcloud'], use_column_width=True)
                st.caption("Visual representation of most frequent terms in the meeting")
            else:
                st.info("Could not generate word cloud from the text")
        
        with col2:
            st.markdown("#### 📈 Word Frequency Distribution")
            if viz_data['frequency_chart']:
                st.image(viz_data['frequency_chart'], use_column_width=True)
                st.caption("Top 20 most frequently used words in the meeting")
            else:
                st.info("Could not generate frequency chart from the text")
        
        # Text Statistics
        st.markdown("#### 📋 Text Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Total Words", viz_data['statistics']['total_words'])
        
        with stats_col2:
            st.metric("Unique Words", viz_data['statistics']['unique_words'])
        
        with stats_col3:
            st.metric("Sentences", viz_data['statistics']['sentences'])
        
        with stats_col4:
            st.metric("Avg. Sentence Length", f"{viz_data['statistics']['avg_sentence_length']:.1f}")
        
        # Most Common Words Table
        st.markdown("#### 🏆 Top 10 Most Frequent Words")
        if viz_data['statistics']['most_common_words']:
            common_words_data = {
                'Word': [word for word, count in viz_data['statistics']['most_common_words']],
                'Frequency': [count for word, count in viz_data['statistics']['most_common_words']]
            }
            st.dataframe(common_words_data, use_container_width=True)
    else:
        st.warning("Not enough text data to generate meaningful visualizations")

def clean_text_for_display(text):
    """Clean and split text for better display"""
    import re
    # Remove "So, " patterns at the beginning
    text = re.sub(r'^So,\s+', '', text.strip())
    # Split by sentences or newlines
    sentences = re.split(r'[.!?]\s+|\n', text)
    # Filter out empty strings and very short sentences
    cleaned = [s.strip() for s in sentences if len(s.strip()) > 10]
    return cleaned

def process_audio_file(uploaded_file, transcription_model, summarization_model):
    """Process the uploaded audio file and generate meeting notes"""
    
    try:
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Save and validate audio file
        status_text.text("Step 1/5: Processing audio file...")
        audio_processor = AudioProcessor()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_audio_path = tmp_file.name
        
        # Validate and get audio info
        audio_info = audio_processor.get_audio_info(temp_audio_path)
        progress_bar.progress(20)
        
        # Display audio info
        st.markdown("#### 🎵 Audio Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{audio_info['duration']:.2f} seconds")
        with col2:
            st.metric("Sample Rate", f"{audio_info['sample_rate']} Hz")
        with col3:
            st.metric("Channels", audio_info['channels'])
        
        # Step 2: Transcribe audio
        status_text.text("Step 2/5: Transcribing audio...")
        transcript_generator = TranscriptGenerator(model_name=transcription_model)
        
        with st.spinner("🎙️ Transcribing audio content... This may take a few moments."):
            transcript = transcript_generator.transcribe_audio(temp_audio_path)
        
        progress_bar.progress(40)
        
        # Step 3: Generate summary
        status_text.text("Step 3/5: Generating summary...")
        summary_generator = SummaryGenerator(model_name=summarization_model)
        
        with st.spinner("📊 Analyzing and summarizing content... Extracting key insights."):
            summary_result = summary_generator.generate_summary(transcript)
        
        progress_bar.progress(60)
        
        # Step 4: Display results in full screen
        status_text.text("Step 4/5: Finalizing results...")
        
        st.markdown("---")
        display_results_fullscreen(transcript, summary_result)
        
        progress_bar.progress(80)
        
        # Step 5: Generate and display visualizations (NEW STEP)
        status_text.text("Step 5/5: Generating text insights...")
        display_visualizations(transcript)
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Download section
        st.markdown("---")
        st.markdown("#### 💾 Download Results")
        
        col1, col2, col3 = st.columns(3)  # Updated to 3 columns
        
        with col1:
            # Download transcript
            st.download_button(
                label="📥 Download Transcript",
                data=transcript,
                file_name=f"meeting_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Download summary
            summary_text = format_summary_for_download(summary_result)
            st.download_button(
                label="📥 Download Summary",
                data=summary_text,
                file_name=f"meeting_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            # Download statistics (NEW)
            stats_text = format_statistics_for_download(transcript)
            st.download_button(
                label="📊 Download Statistics",
                data=stats_text,
                file_name=f"meeting_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Cleanup
        os.unlink(temp_audio_path)
        
        st.success("✅ Meeting notes generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Error processing audio file: {str(e)}")
        st.info("💡 Please try again with a different audio file or check the file format.")

def format_summary_for_download(summary_result):
    """Format summary result for download"""
    download_text = f"""MEETING SUMMARY
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
{summary_result['overall_summary']}

KEY DISCUSSION POINTS:
"""
    
    if summary_result["key_points"] and summary_result["key_points"] != "No key points identified.":
        key_points_list = clean_text_for_display(summary_result["key_points"])
        for i, point in enumerate(key_points_list, 1):
            if point.strip():
                download_text += f"{i}. {point.strip()}\n"
    else:
        download_text += "No key points identified.\n"
    
    download_text += "\nACTION ITEMS:\n"
    if summary_result["action_items"] and summary_result["action_items"] != "No specific action items identified.":
        action_points = clean_text_for_display(summary_result["action_items"])
        for action in action_points:
            if action.strip():
                download_text += f"[ ] {action.strip()}\n"
    else:
        download_text += "No specific action items identified.\n"
    
    download_text += "\nDECISIONS MADE:\n"
    if summary_result["decisions"] and summary_result["decisions"] != "No specific decisions identified.":
        decision_points = clean_text_for_display(summary_result["decisions"])
        for decision in decision_points:
            if decision.strip():
                download_text += f"✓ {decision.strip()}\n"
    else:
        download_text += "No specific decisions identified.\n"
    
    download_text += "\n---\nGenerated by Intelligent Meeting Notes Generator"
    
    return download_text

def format_statistics_for_download(text):
    """Format text statistics for download"""
    visualizer = TextVisualizer()
    stats = visualizer.generate_text_statistics(text)
    
    stats_text = f"""MEETING TEXT STATISTICS
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BASIC STATISTICS:
• Total Words: {stats['total_words']}
• Unique Words: {stats['unique_words']}
• Sentences: {stats['sentences']}
• Average Sentence Length: {stats['avg_sentence_length']:.1f} words

TOP 10 MOST FREQUENT WORDS:
"""
    
    for word, count in stats['most_common_words']:
        stats_text += f"• {word}: {count} occurrences\n"
    
    stats_text += "\n---\nGenerated by Intelligent Meeting Notes Generator"
    
    return stats_text

def about_project():
    """About project section with interactive flip cards"""
    
    # Main header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem; color: #333; margin-bottom: 1rem;">About Intelligent Meeting Notes Generator</h1>
        <p style="font-size: 1.2rem; color: #666; max-width: 800px; margin: 0 auto;">
        Transform your meeting audio into structured, actionable notes using state-of-the-art AI technologies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Custom CSS for flip cards
    st.markdown("""
    <style>
    .flip-card {
        background-color: transparent;
        width: 100%;
        height: 200px;
        perspective: 1000px;
        margin: 15px 0;
        border-radius: 15px;
    }
    
    .flip-card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.6s;
        transform-style: preserve-3d;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-radius: 15px;
    }
    
    .flip-card:hover .flip-card-inner {
        transform: rotateY(180deg);
    }
    
    .flip-card-front, .flip-card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        border-radius: 15px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 20px;
        box-sizing: border-box;
    }
    
    .flip-card-front {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .flip-card-back {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        transform: rotateY(180deg);
    }
    
    .tech-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .tech-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .card-content {
        font-size: 0.95rem;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Features Section with Flip Cards
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #333; margin-bottom: 1rem;">🚀 Key Features</h2>
        <p style="color: #666;">Hover over each card to learn more</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards in 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Card 1: Audio Transcription
        st.markdown("""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <div class="feature-icon">🎙️</div>
                    <div class="card-title">Audio Transcription</div>
                    <div class="card-content">Convert speech to text</div>
                </div>
                <div class="flip-card-back">
                    <div class="card-title">Powered by OpenAI Whisper</div>
                    <div class="card-content">
                        Advanced speech recognition that converts meeting audio into accurate text transcripts with support for multiple languages and accents.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Card 2: Intelligent Summarization
        st.markdown("""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <div class="feature-icon">📝</div>
                    <div class="card-title">Smart Summarization</div>
                    <div class="card-content">Extract key insights</div>
                </div>
                <div class="flip-card-back">
                    <div class="card-title">NLP Algorithms</div>
                    <div class="card-content">
                        Uses advanced Natural Language Processing to generate concise summaries, highlighting the most important discussion points.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Card 3: Action Item Extraction
        st.markdown("""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <div class="feature-icon">✅</div>
                    <div class="card-title">Action Items</div>
                    <div class="card-content">Track tasks & responsibilities</div>
                </div>
                <div class="flip-card-back">
                    <div class="card-title">Automatic Task Detection</div>
                    <div class="card-content">
                        Identifies and extracts action items, assignments, and deadlines mentioned during meetings for better follow-up.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Card 4: Decision Tracking
        st.markdown("""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <div class="feature-icon">🎯</div>
                    <div class="card-title">Decision Tracking</div>
                    <div class="card-content">Capture important decisions</div>
                </div>
                <div class="flip-card-back">
                    <div class="card-title">Decision Recognition</div>
                    <div class="card-content">
                        Automatically detects and highlights key decisions made during meetings for clear documentation and reference.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Card 5: Key Points Identification
        st.markdown("""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <div class="feature-icon">📊</div>
                    <div class="card-title">Key Points</div>
                    <div class="card-content">Highlight main topics</div>
                </div>
                <div class="flip-card-back">
                    <div class="card-title">Important Topic Extraction</div>
                    <div class="card-content">
                        Identifies and organizes the most critical discussion points, ensuring nothing important gets missed.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Card 6: Export Functionality
        st.markdown("""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <div class="feature-icon">💾</div>
                    <div class="card-title">Export Results</div>
                    <div class="card-content">Download in multiple formats</div>
                </div>
                <div class="flip-card-back">
                    <div class="card-title">Flexible Export Options</div>
                    <div class="card-content">
                        Export transcripts and summaries as text files for easy sharing and integration with other tools.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology Stack Section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #333; margin-bottom: 1rem;">🛠️ Technology Stack</h2>
        <p style="color: #666;">Modern technologies powering our solution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology cards in 3 columns
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        <div class="tech-card">
            <div class="feature-icon">🎨</div>
            <div class="card-title">Frontend</div>
            <div class="card-content">
                <strong>Streamlit</strong><br>
                Interactive web application framework for creating beautiful data apps in pure Python.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-card">
            <div class="feature-icon">🔊</div>
            <div class="card-title">Audio Processing</div>
            <div class="card-content">
                <strong>Librosa & Pydub</strong><br>
                Advanced audio processing libraries for handling various audio formats and preprocessing.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown("""
        <div class="tech-card">
            <div class="feature-icon">🐍</div>
            <div class="card-title">Backend</div>
            <div class="card-content">
                <strong>Python</strong><br>
                Powerful programming language with extensive libraries for AI and data processing.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-card">
            <div class="feature-icon">📚</div>
            <div class="card-title">Text Processing</div>
            <div class="card-content">
                <strong>NLTK & Sumy</strong><br>
                Natural Language Toolkit and text summarization libraries for intelligent text analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        st.markdown("""
        <div class="tech-card">
            <div class="feature-icon">🎙️</div>
            <div class="card-title">Speech Recognition</div>
            <div class="card-content">
                <strong>OpenAI Whisper</strong><br>
                State-of-the-art automatic speech recognition system for accurate transcription.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-card">
            <div class="feature-icon">🧠</div>
            <div class="card-title">Machine Learning</div>
            <div class="card-content">
                <strong>PyTorch</strong><br>
                Deep learning framework powering the AI models for speech and text processing.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How It Works Section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #333; margin-bottom: 1rem;">📊 How It Works</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Process steps
    steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
    
    with steps_col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 3rem; margin-bottom: 15px;">1️⃣</div>
            <h4 style="color: #333; margin-bottom: 10px;">Upload Audio</h4>
            <p style="color: #666;">Upload your meeting recording in supported formats</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 3rem; margin-bottom: 15px;">2️⃣</div>
            <h4 style="color: #333; margin-bottom: 10px;">Transcribe</h4>
            <p style="color: #666;">AI converts speech to accurate text</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 3rem; margin-bottom: 15px;">3️⃣</div>
            <h4 style="color: #333; margin-bottom: 10px;">Analyze</h4>
            <p style="color: #666;">NLP extracts key insights and patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col4:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 3rem; margin-bottom: 15px;">4️⃣</div>
            <h4 style="color: #333; margin-bottom: 10px;">Summarize</h4>
            <p style="color: #666;">Get structured, actionable meeting notes</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Get settings from sidebar
    transcription_model, summarization_model, app_mode = main()
    
    if app_mode == "Upload Audio":
        # Show upload interface
        uploaded_file, trans_model, summ_model = upload_audio_interface(transcription_model, summarization_model)
        
        if uploaded_file is not None:
            # Process the file
            process_audio_file(uploaded_file, trans_model, summ_model)
    else:
        about_project()
