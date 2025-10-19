import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import io
import base64

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextVisualizer:
    """Handles text visualization including word clouds and frequency analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add custom stop words for meeting context
        self.custom_stop_words = {
            'so', 'okay', 'well', 'like', 'you know', 'i mean', 'actually', 
            'basically', 'uh', 'um', 'ah', 'yes', 'no', 'maybe', 'please',
            'thank', 'thanks', 'hello', 'hi', 'hey', 'meeting', 'discuss',
            'discussion', 'team', 'project', 'today'
        }
        self.stop_words.update(self.custom_stop_words)
    
    def preprocess_text_for_visualization(self, text):
        """Clean and preprocess text for visualization"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stop words and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return filtered_words
    
    def generate_wordcloud(self, text, width=800, height=400):
        """Generate word cloud from text and return as base64 image"""
        try:
            # Preprocess text
            words = self.preprocess_text_for_visualization(text)
            processed_text = ' '.join(words)
            
            if not processed_text.strip():
                return None
            
            # Create word cloud
            wordcloud = WordCloud(
                width=width,
                height=height,
                background_color='white',
                colormap='viridis',
                max_words=100,
                contour_width=1,
                contour_color='steelblue',
                relative_scaling=0.5
            ).generate(processed_text)
            
            # Convert to base64 for Streamlit
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud - Most Frequent Meeting Terms', fontsize=16, pad=20)
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generating word cloud: {e}")
            return None
    
    def generate_word_frequency_chart(self, text, top_n=20):
        """Generate word frequency bar chart"""
        try:
            # Preprocess and count words
            words = self.preprocess_text_for_visualization(text)
            word_freq = Counter(words)
            most_common = word_freq.most_common(top_n)
            
            if not most_common:
                return None
            
            # Prepare data for plotting
            words_list = [item[0] for item in most_common]
            counts = [item[1] for item in most_common]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            bars = plt.barh(words_list, counts, color='skyblue', edgecolor='navy')
            
            # Customize plot
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('Words', fontsize=12)
            plt.title(f'Top {top_n} Most Frequent Words', fontsize=16, pad=20)
            plt.gca().invert_yaxis()  # Highest frequency at top
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{int(width)}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error generating frequency chart: {e}")
            return None
    
    def generate_text_statistics(self, text):
        """Generate basic text statistics"""
        words = self.preprocess_text_for_visualization(text)
        sentences = nltk.sent_tokenize(text)
        
        stats = {
            'total_words': len(text.split()),
            'unique_words': len(set(words)),
            'sentences': len(sentences),
            'avg_sentence_length': len(text.split()) / len(sentences) if sentences else 0,
            'most_common_words': Counter(words).most_common(10)
        }
        
        return stats
    
    def create_visualization_section(self, text):
        """Create complete visualization section for Streamlit"""
        if not text or len(text.strip()) < 50:
            return None
        
        # Generate visualizations
        wordcloud_buffer = self.generate_wordcloud(text)
        freq_chart_buffer = self.generate_word_frequency_chart(text)
        stats = self.generate_text_statistics(text)
        
        return {
            'wordcloud': wordcloud_buffer,
            'frequency_chart': freq_chart_buffer,
            'statistics': stats
        }
