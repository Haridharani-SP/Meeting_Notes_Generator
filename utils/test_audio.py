import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from utils.audio_processor import AudioProcessor

def test_audio_processor():
    print("🔧 Testing Audio Processor...")
    
    processor = AudioProcessor()
    
    # Test FFmpeg availability
    print(f"FFmpeg available: {processor.ffmpeg_available}")
    
    # Test with a sample audio file (you'll need to provide one)
    print("\n📝 Please provide the path to a test audio file:")
    audio_path = input("Audio file path: ").strip().strip('"')
    
    if not os.path.exists(audio_path):
        print("❌ File not found!")
        return
    
    try:
        # Test audio info
        print("\n🎵 Getting audio info...")
        info = processor.get_audio_info(audio_path)
        print(f"✅ Audio info: {info}")
        
        # Test conversion
        print("\n🔄 Testing audio conversion...")
        wav_path = processor.convert_to_wav(audio_path)
        print(f"✅ Converted to: {wav_path}")
        
        # Test preprocessing
        print("\n⚡ Testing audio preprocessing...")
        processed_path = processor.preprocess_audio(audio_path)
        print(f"✅ Processed path: {processed_path}")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_audio_processor()
