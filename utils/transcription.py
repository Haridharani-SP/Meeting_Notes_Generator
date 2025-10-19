import whisper
import torch
import tempfile
import os
from .audio_processor import AudioProcessor

class TranscriptGenerator:
    """Handles speech-to-text transcription using OpenAI Whisper directly"""
    
    def __init__(self, model_name="tiny"):
        # Map model names from app to whisper model sizes
        model_map = {
            "openai/whisper-tiny": "tiny",
            "openai/whisper-base": "base", 
            "openai/whisper-small": "small"
        }
        
        self.whisper_model_size = model_map.get(model_name, "base")
        self.model = None
        self.audio_processor = AudioProcessor()
        
    def load_model(self):
        """Load the Whisper model"""
        try:
            print(f"Loading Whisper model: {self.whisper_model_size}")
            self.model = whisper.load_model(self.whisper_model_size)
            print("✅ Whisper model loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading Whisper model: {str(e)}")
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file to text using Whisper"""
        try:
            if self.model is None:
                self.load_model()
            
            # Preprocess audio
            processed_audio_path = self.audio_processor.preprocess_audio(audio_path)
            
            # Perform transcription with Whisper
            print("Starting transcription with Whisper...")
            result = self.model.transcribe(
                processed_audio_path,
                fp16=False  # Use FP32 for better compatibility
            )
            
            # Cleanup processed audio file
            if processed_audio_path != audio_path:
                os.unlink(processed_audio_path)
            
            transcript = result["text"]
            
            # Clean and format transcript
            cleaned_transcript = self.clean_transcript(transcript)
            
            return cleaned_transcript
            
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
    
    def clean_transcript(self, transcript):
        """Clean and format the transcript text"""
        import re
        
        # Remove extra whitespace
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        
        # Basic punctuation restoration
        transcript = re.sub(r'(\w)([.!?])(\w)', r'\1\2 \3', transcript)
        
        # Capitalize first letter
        if transcript:
            transcript = transcript[0].upper() + transcript[1:]
        
        return transcript
