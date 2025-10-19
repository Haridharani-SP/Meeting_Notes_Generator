import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SummaryGenerator:
    """Handles text summarization using traditional NLP methods"""
    
    def __init__(self, model_name="lsa"):
        self.model_name = model_name
        self.stop_words = set(stopwords.words('english'))
        
    def generate_summary(self, text, sentences_count=5):
        """Generate comprehensive meeting summary using traditional NLP"""
        try:
            # Use Sumy for summarization (no transformers dependency)
            if len(text.split()) < 100:
                # For very short texts, use extractive methods
                summary = self.extractive_summarization(text, sentences_count)
            else:
                # For longer texts, use multiple methods
                summary = self.multi_method_summarization(text, sentences_count)
            
            # Extract structured information
            structured_result = self.extract_structured_info(text, summary)
            
            return structured_result
            
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")
    
    def extractive_summarization(self, text, sentences_count=5):
        """Use Sumy library for extractive summarization"""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            
            if self.model_name == "textrank":
                summarizer = TextRankSummarizer()
            else:  # Default to LSA
                summarizer = LsaSummarizer()
            
            summary_sentences = summarizer(parser.document, sentences_count)
            summary = " ".join(str(sentence) for sentence in summary_sentences)
            
            return summary if summary else self.fallback_summary(text, sentences_count)
            
        except Exception:
            return self.fallback_summary(text, sentences_count)
    
    def multi_method_summarization(self, text, sentences_count=5):
        """Combine multiple summarization methods"""
        try:
            # Method 1: Sumy LSA
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            lsa_summarizer = LsaSummarizer()
            lsa_summary = lsa_summarizer(parser.document, sentences_count)
            
            # Method 2: TextRank
            textrank_summarizer = TextRankSummarizer()
            textrank_summary = textrank_summarizer(parser.document, sentences_count)
            
            # Combine both methods
            combined_sentences = list(lsa_summary) + list(textrank_summary)
            unique_sentences = list(dict.fromkeys(str(s) for s in combined_sentences))
            
            summary = " ".join(unique_sentences[:sentences_count])
            
            return summary if summary else self.fallback_summary(text, sentences_count)
            
        except Exception:
            return self.fallback_summary(text, sentences_count)
    
    def fallback_summary(self, text, sentences_count=3):
        """Fallback summary using sentence scoring"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= sentences_count:
            return text
        
        # Simple scoring based on sentence length and keywords
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            # Score based on length and important keywords
            score = len(words) + sum(1 for word in words if word in ['important', 'decision', 'action', 'need', 'must', 'will'])
            scored_sentences.append((score, sentence))
        
        # Get top sentences
        scored_sentences.sort(reverse=True)
        top_sentences = [sentence for _, sentence in scored_sentences[:sentences_count]]
        
        return " ".join(top_sentences)
    
    def extract_structured_info(self, full_text, summary):
        """Extract action items, decisions, and key points using regex patterns"""
        
        # Extract action items
        action_items = self.extract_action_items(full_text)
        
        # Extract decisions
        decisions = self.extract_decisions(full_text)
        
        # Extract key points (most important sentences)
        key_points = self.extract_key_points(full_text)
        
        return {
            "overall_summary": summary,
            "action_items": action_items,
            "decisions": decisions,
            "key_points": key_points
        }
    
    def extract_action_items(self, text):
        """Extract action items using pattern matching"""
        action_patterns = [
            r"(\b[Ww]e\s+(?:need to|should|must|will)\s+[^.!?]+[.!?])",
            r"(\b[Ii]'ll\s+[^.!?]+[.!?])",
            r"(\b[Aa]ction\s+[^.!?]+[.!?])",
            r"(\b[Ll]et'?s\s+[^.!?]+[.!?])",
            r"(\b[Nn]ext\s+steps?\s*[^.!?]*[.!?])",
            r"(\b[Tt]o\s+do\s*[^.!?]*[.!?])"
        ]
        
        action_items = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text)
            action_items.extend(matches)
        
        return "\n".join(action_items[:5]) if action_items else "No specific action items identified."
    
    def extract_decisions(self, text):
        """Extract decisions using pattern matching"""
        decision_patterns = [
            r"(\b[Dd]ecided\s+[^.!?]+[.!?])",
            r"(\b[Aa]greed\s+[^.!?]+[.!?])",
            r"(\b[Ww]e\s+will\s+[^.!?]+[.!?])",
            r"(\b[Ff]inalized\s+[^.!?]+[.!?])",
            r"(\b[Rr]esolved\s+[^.!?]+[.!?])"
        ]
        
        decisions = []
        for pattern in decision_patterns:
            matches = re.findall(pattern, text)
            decisions.extend(matches)
        
        return "\n".join(decisions[:5]) if decisions else "No specific decisions identified."
    
    def extract_key_points(self, text):
        """Extract key discussion points"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 5:
            return "\n".join(sentences)
        
        # Simple heuristic: longer sentences often contain more information
        key_sentences = sorted(sentences, key=len, reverse=True)[:5]
        
        return "\n".join(key_sentences)
    
    def clean_meeting_text(self, text):
        """Clean meeting text by removing fillers and repetitions"""
        import re
        
        # Remove common filler words at the beginning of sentences
        fillers = ['so', 'okay', 'well', 'like', 'you know', 'i mean', 'actually', 'basically']
        pattern = r'\b(' + '|'.join(fillers) + r')\b[, ]*'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove repeated phrases
        text = re.sub(r'(\b\w+\b)( \1)+', r'\1', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
    
        return text
