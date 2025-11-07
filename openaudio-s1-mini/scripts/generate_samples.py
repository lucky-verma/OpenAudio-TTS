"""
Script to generate TTS samples with different configurations.
This script helps create diverse TTS samples for testing and demonstration.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import fish_speech modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
import soundfile as sf
from datetime import datetime


def setup_model(device="auto", checkpoint_path="checkpoints/openaudio-s1-mini"):
    """Initialize the TTS model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model on {device}...")
    
    # Load models
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=checkpoint_path,
        device=device,
        precision=torch.bfloat16,
        compile=True,
    )
    
    decoder_model = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path=f"{checkpoint_path}/codec.pth",
        device=device,
    )
    
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=True,
        precision=torch.bfloat16,
    )
    
    print("Model loaded successfully!")
    return inference_engine


def generate_sample(
    inference_engine,
    text,
    output_path,
    reference_audio=None,
    reference_text=None,
    emotion=None,
    **kwargs
):
    """Generate a single TTS sample."""
    # Add emotion marker if specified
    if emotion:
        text = f"({emotion}) {text}"
    
    # Prepare request
    request = ServeTTSRequest(
        text=text,
        references=[] if reference_audio is None else [reference_audio],
        reference_id=None,
        max_new_tokens=kwargs.get("max_new_tokens", 1024),
        chunk_length=kwargs.get("chunk_length", 200),
        top_p=kwargs.get("top_p", 0.9),
        repetition_penalty=kwargs.get("repetition_penalty", 1.1),
        temperature=kwargs.get("temperature", 0.9),
        format="wav",
    )
    
    # Generate audio
    print(f"Generating: {text[:50]}...")
    audio_chunks = list(inference_engine.inference(request))
    
    # Concatenate chunks
    if len(audio_chunks) > 1:
        import numpy as np
        audio = np.concatenate([chunk["audio"] for chunk in audio_chunks])
        sample_rate = audio_chunks[0]["sample_rate"]
    else:
        audio = audio_chunks[0]["audio"]
        sample_rate = audio_chunks[0]["sample_rate"]
    
    # Save audio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    print(f"Saved: {output_path}")
    
    return output_path


def main():
    """Generate sample TTS outputs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate TTS samples")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/openaudio-s1-mini")
    parser.add_argument("--output-dir", type=str, default="samples/outputs")
    args = parser.parse_args()
    
    # Setup model
    inference_engine = setup_model(args.device, args.checkpoint_path)
    
    # Sample texts with different emotions
    samples = [
        {
            "text": "Hello, this is a basic text-to-speech sample.",
            "emotion": None,
            "filename": "basic_sample.wav"
        },
        {
            "text": "I am feeling very excited about this new technology!",
            "emotion": "excited",
            "filename": "excited_sample.wav"
        },
        {
            "text": "This is a sad and melancholic sentence.",
            "emotion": "sad",
            "filename": "sad_sample.wav"
        },
        {
            "text": "I am angry about this situation!",
            "emotion": "angry",
            "filename": "angry_sample.wav"
        },
        {
            "text": "This is being whispered very quietly.",
            "emotion": "whispering",
            "filename": "whisper_sample.wav"
        },
        {
            "text": "I am shouting at the top of my lungs!",
            "emotion": "shouting",
            "filename": "shouting_sample.wav"
        },
        {
            "text": "Hello world! 你好世界! こんにちは世界!",
            "emotion": None,
            "filename": "multilingual_sample.wav"
        },
    ]
    
    # Generate samples
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {len(samples)} samples...")
    print(f"Output directory: {output_dir}\n")
    
    for i, sample in enumerate(samples, 1):
        output_path = output_dir / sample["filename"]
        try:
            generate_sample(
                inference_engine,
                sample["text"],
                str(output_path),
                emotion=sample["emotion"]
            )
            print(f"[{i}/{len(samples)}] ✓\n")
        except Exception as e:
            print(f"[{i}/{len(samples)})] ✗ Error: {e}\n")
    
    print(f"\nAll samples generated in: {output_dir}")


if __name__ == "__main__":
    main()

