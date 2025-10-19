import os
import subprocess
import sys
import librosa
import soundfile as sf
import tempfile
import shutil

# Add FFmpeg to system PATH for Python
ffmpeg_paths = [
    r"C:\Users\DELL\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin",
    r"C:\ffmpeg\bin", 
    r"C:\Program Files\FFmpeg\bin"
]

for path in ffmpeg_paths:
    if os.path.exists(path):
        os.environ["PATH"] += os.pathsep + path
        break

# Set FFmpeg path for pydub
for path in ffmpeg_paths:
    ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
    if os.path.exists(ffmpeg_exe):
        os.environ["FFMPEG_PATH"] = ffmpeg_exe
        break

class AudioProcessor:
    """Handles audio file processing with multiple fallback methods"""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac']
        self.ffmpeg_available = self.check_ffmpeg()
    
    def check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            print("⚠️ FFmpeg not available, using fallback methods")
            return False
    
    def get_audio_info(self, audio_path):
        """Get basic information about the audio file"""
        try:
            # Load audio file with librosa (handles most formats without FFmpeg)
            audio, sample_rate = librosa.load(audio_path, sr=None, mono=False)
            
            # Handle multi-channel audio
            if len(audio.shape) > 1:
                channels = audio.shape[0]
                duration = audio.shape[1] / sample_rate
            else:
                channels = 1
                duration = len(audio) / sample_rate
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'samples': len(audio) if channels == 1 else audio.shape[1]
            }
        except Exception as e:
            raise Exception(f"Error reading audio file: {str(e)}")
    
    def convert_to_wav(self, input_path, output_path=None):
        """Convert audio file to WAV format using multiple methods"""
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.wav')
            
            # Method 1: Try librosa + soundfile (no FFmpeg needed)
            try:
                print("🔄 Converting audio using librosa...")
                y, sr = librosa.load(input_path, sr=16000, mono=True)
                sf.write(output_path, y, sr)
                print("✅ Audio converted successfully with librosa")
                return output_path
            except Exception as e:
                print(f"❌ Librosa conversion failed: {e}")
            
            # Method 2: Try FFmpeg if available
            if self.ffmpeg_available:
                try:
                    print("🔄 Converting audio using FFmpeg...")
                    cmd = [
                        'ffmpeg', '-i', input_path, 
                        '-ac', '1', '-ar', '16000', 
                        '-y', output_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        print("✅ Audio converted successfully with FFmpeg")
                        return output_path
                    else:
                        print(f"❌ FFmpeg conversion failed: {result.stderr}")
                except Exception as e:
                    print(f"❌ FFmpeg conversion failed: {e}")
            
            # Method 3: Direct copy if already WAV
            if input_path.lower().endswith('.wav'):
                shutil.copy2(input_path, output_path)
                return output_path
            
            raise Exception("All conversion methods failed")
            
        except Exception as e:
            raise Exception(f"Error converting audio to WAV: {str(e)}")
    
    def validate_audio_file(self, file_path):
        """Validate if the audio file is processable"""
        try:
            info = self.get_audio_info(file_path)
            
            # Check duration
            if info['duration'] < 1.0:
                raise ValueError("Audio file is too short (less than 1 second)")
            
            if info['duration'] > 3600:  # 1 hour
                raise ValueError("Audio file is too long (more than 1 hour)")
            
            return True
            
        except Exception as e:
            raise Exception(f"Audio file validation failed: {str(e)}")
    
    def preprocess_audio(self, input_path):
        """Preprocess audio for better transcription - SIMPLIFIED VERSION"""
        try:
            print(f"🔄 Preprocessing audio: {input_path}")
            
            # For now, just convert to WAV and return the same path
            # This avoids complex processing that might fail
            if not input_path.lower().endswith('.wav'):
                wav_path = self.convert_to_wav(input_path)
                return wav_path
            else:
                # If it's already WAV, just return the same path
                return input_path
            
        except Exception as e:
            raise Exception(f"Audio preprocessing failed: {str(e)}")
