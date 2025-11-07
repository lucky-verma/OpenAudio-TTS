#!/usr/bin/env python3
"""
Voice Cloning Test Script

Tests voice cloning with reference audio across multiple languages.
Uses Recording.m4a as reference audio and generates TTS in 8 languages.

Usage:
    python3 test_voice_cloning.py

Output:
    Generated WAV files saved to ~/openaudio_test_outputs/
"""

import requests
import base64
import os
import sys

# API endpoints
S1_MINI_URL = "http://localhost:8080/v1"
FISH_15_URL = "http://localhost:8081/v1"

# Reference audio path (relative to docker folder or absolute)
REF_AUDIO_PATH = "../Recording.m4a"

# Output directory (use home directory for write permissions)
OUTPUT_DIR = os.path.expanduser("~/openaudio_test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_audio_to_base64(audio_path):
    """Read audio file and encode to base64"""
    if not os.path.exists(audio_path):
        # Try absolute path
        abs_path = os.path.abspath(audio_path)
        if os.path.exists(abs_path):
            audio_path = abs_path
        else:
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    file_size = len(audio_bytes)
    print(f"   ✓ Loaded reference audio: {audio_path}")
    print(f"   Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    return audio_b64

def test_voice_cloning(url, name, audio_b64, text, reference_text, output_file):
    """Test voice cloning with given text"""
    print(f"\n{name} - Generating: '{text[:50]}...'")
    try:
        response = requests.post(
            f"{url}/tts",
            json={
                "text": text,
                "references": [{
                    "audio": audio_b64,
                    "text": reference_text
                }],
                "temperature": 0.9,
                "top_p": 0.9,
                "format": "wav"
            },
            timeout=180  # Voice cloning can take longer
        )
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_file)
        print(f"   ✓ Generated: {output_file}")
        print(f"   Size: {file_size:,} bytes")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text[:200]}")
        return False

def test_multilingual_examples(url, name, audio_b64, reference_text):
    """Test multilingual examples with voice cloning"""
    print(f"\n{'='*60}")
    print(f"{name} - Multilingual Voice Cloning Tests")
    print(f"{'='*60}")
    
    # Multilingual test cases
    test_cases = [
        {
            "language": "English",
            "text": "Hello world! This is a test of voice cloning in English.",
            "reference_text": reference_text,
            "output": os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_en.wav")
        },
        {
            "language": "Chinese (Simplified)",
            "text": "你好世界！这是一个中文语音克隆测试。",
            "reference_text": reference_text,
            "output": os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_zh.wav")
        },
        {
            "language": "Japanese",
            "text": "こんにちは世界！これは日本語の音声クローニングテストです。",
            "reference_text": reference_text,
            "output": os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_ja.wav")
        },
        {
            "language": "Korean",
            "text": "안녕하세요 세계! 이것은 한국어 음성 복제 테스트입니다.",
            "reference_text": reference_text,
            "output": os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_ko.wav")
        },
        {
            "language": "Spanish",
            "text": "¡Hola mundo! Esta es una prueba de clonación de voz en español.",
            "reference_text": reference_text,
            "output": os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_es.wav")
        },
        {
            "language": "French",
            "text": "Bonjour le monde! Ceci est un test de clonage vocal en français.",
            "reference_text": reference_text,
            "output": os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_fr.wav")
        },
        {
            "language": "German",
            "text": "Hallo Welt! Dies ist ein Test zur Stimmklonierung auf Deutsch.",
            "reference_text": reference_text,
            "output": os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_de.wav")
        },
        {
            "language": "Mixed (English + Chinese)",
            "text": "Hello 世界! This is a 混合 language test with 中文 and English.",
            "reference_text": reference_text,
            "output": os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_mixed.wav")
        }
    ]
    
    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test['language']}:")
        success = test_voice_cloning(
            url, 
            name,
            audio_b64,
            test['text'],
            test['reference_text'],
            test['output']
        )
        results.append((test['language'], success))
    
    return results

def check_service(url, name):
    """Check if service is available"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def main():
    print("=" * 60)
    print("Voice Cloning Test with Multilingual Examples")
    print("=" * 60)
    
    # Check which services are available
    s1_available = check_service(S1_MINI_URL, "OpenAudio S1 Mini")
    fish_available = check_service(FISH_15_URL, "Fish Speech 1.5")
    
    if not s1_available and not fish_available:
        print("\n✗ No services available!")
        print("  Start containers with: docker-compose up -d")
        sys.exit(1)
    
    # Load reference audio
    print("\n" + "=" * 60)
    print("Loading Reference Audio")
    print("=" * 60)
    try:
        audio_b64 = encode_audio_to_base64(REF_AUDIO_PATH)
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease provide the reference audio file:")
        print(f"  Expected location: {os.path.abspath(REF_AUDIO_PATH)}")
        print("\nOr update REF_AUDIO_PATH in the script.")
        sys.exit(1)
    
    # Reference text (you can update this with the actual transcription)
    reference_text = "This is the reference audio for voice cloning."
    
    # Test available services
    all_results = []
    
    if s1_available:
        results = test_multilingual_examples(
            S1_MINI_URL, 
            "OpenAudio S1 Mini",
            audio_b64,
            reference_text
        )
        all_results.extend([(f"S1 Mini - {lang}", result) for lang, result in results])
    
    if fish_available:
        results = test_multilingual_examples(
            FISH_15_URL,
            "Fish Speech 1.5",
            audio_b64,
            reference_text
        )
        all_results.extend([(f"Fish 1.5 - {lang}", result) for lang, result in results])
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    for name, result in all_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All voice cloning tests passed!")
        print("\nGenerated files:")
        for name, result in all_results:
            if result:
                # Extract filename from test name
                parts = name.split(" - ")
                if len(parts) == 2:
                    lang = parts[1].lower().replace(" ", "_")
                    service = parts[0].lower().replace(" ", "_")
                    print(f"  - {service}_{lang}.wav")
    else:
        print("\n⚠️  Some tests failed. Check container logs:")
        print("   docker-compose logs")

if __name__ == "__main__":
    main()

