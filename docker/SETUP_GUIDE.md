# OpenAudio S1 Mini & Fish Speech 1.5 - Complete Setup Guide

**Models:** OpenAudio S1-mini (0.5B) and Fish Speech 1.5 (500M)  
**Languages:** EN, ZH, JA, KO, ES, FR, DE, AR + more  
**Requirements:** Docker, NVIDIA GPU with 12GB+ VRAM, Hugging Face CLI

---

## STEP 1: Install Hugging Face CLI (one-time setup)

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
source ~/.bashrc
```

---

## STEP 2: Download Model Weights

```bash
cd ~
mkdir -p fish-speech-models
cd fish-speech-models

# Download OpenAudio S1 Mini (3.6GB - recommended)
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# Download Fish Speech 1.5 (2.1GB - optional)
hf download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

---

## STEP 3: Pull Docker Image

```bash
docker pull fishaudio/fish-speech:server-cuda
```

---

## STEP 4: Run OpenAudio S1 Mini (Primary - Best Quality)

```bash
docker run -d \
    --name openaudio-s1-mini \
    --gpus all \
    -p 8080:8080 \
    -v "$(pwd)/checkpoints":/app/checkpoints \
    -e COMPILE=1 \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini \
    -e DECODER_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini/codec.pth \
    -e DECODER_CONFIG_NAME=modded_dac_vq \
    fishaudio/fish-speech:server-cuda

# Check logs
docker logs -f openaudio-s1-mini
# Wait for: "Application startup complete" then Ctrl+C
```

---

## STEP 5: Run Fish Speech 1.5 (Alternative - Faster)

```bash
# Stop S1 Mini first if running
docker stop openaudio-s1-mini

# Start Fish Speech 1.5 on same port
docker run -d \
    --name fish-speech-1-5 \
    --gpus all \
    -p 8080:8080 \
    -v "$(pwd)/checkpoints":/app/checkpoints \
    -e COMPILE=1 \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/fish-speech-1.5 \
    fishaudio/fish-speech:server-cuda

# Check logs
docker logs -f fish-speech-1-5
```

---

## STEP 6: Run Both Models Simultaneously (Different Ports)

```bash
# OpenAudio S1 Mini on port 8080
docker run -d \
    --name openaudio-s1-mini \
    --gpus all \
    -p 8080:8080 \
    -v "$(pwd)/checkpoints":/app/checkpoints \
    -e COMPILE=1 \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini \
    fishaudio/fish-speech:server-cuda

# Fish Speech 1.5 on port 8081
docker run -d \
    --name fish-speech-1-5 \
    --gpus all \
    -p 8081:8080 \
    -v "$(pwd)/checkpoints":/app/checkpoints \
    -e COMPILE=1 \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/fish-speech-1.5 \
    fishaudio/fish-speech:server-cuda
```

**Or use Docker Compose:**

```bash
cd docker
docker-compose up -d
```

---

## API Usage Examples

### Example 1: Basic TTS (No Voice Cloning)

#### English

```bash
curl -X POST "http://localhost:8080/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world, this is a test.",
    "format": "wav",
    "normalize": true,
    "temperature": 0.7
  }' \
  --output output_english.wav
```

#### Chinese

```bash
curl -X POST "http://localhost:8080/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好世界，这是一个测试。",
    "format": "wav",
    "normalize": true
  }' \
  --output output_chinese.wav
```

#### Korean

```bash
curl -X POST "http://localhost:8080/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 이것은 테스트입니다.",
    "format": "wav",
    "normalize": true
  }' \
  --output output_korean.wav
```

#### Spanish

```bash
curl -X POST "http://localhost:8080/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hola mundo, esto es una prueba.",
    "format": "wav",
    "normalize": true
  }' \
  --output output_spanish.wav
```

---

### Example 2: Voice Cloning with Reference Audio

#### Step 2a: Add reference voice with ID

```bash
curl -X POST "http://localhost:8080/v1/references/add" \
  -F "id=john_voice" \
  -F "audio=@/path/to/john_sample.wav" \
  -F "text=This is the text spoken in the reference audio file"
```

#### Step 2b: Generate TTS using the reference voice

```bash
curl -X POST "http://localhost:8080/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "New text to synthesize with Johns voice",
    "reference_id": "john_voice",
    "format": "wav",
    "normalize": true,
    "temperature": 0.7
  }' \
  --output cloned_john.wav
```

#### Step 2c: List all reference voices

```bash
curl -X GET "http://localhost:8080/v1/references/list"
```

#### Step 2d: Delete reference voice

```bash
curl -X DELETE "http://localhost:8080/v1/references/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "reference_id": "john_voice"
  }'
```

---

### Example 3: Voice Cloning with Inline Reference (No ID)

```bash
curl -X POST "http://localhost:8080/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is new text with cloned voice",
    "references": [
      {
        "audio": "/path/to/reference.wav",
        "text": "Reference text spoken in the audio"
      }
    ],
    "format": "wav",
    "normalize": true
  }' \
  --output inline_clone.wav
```

---

### Example 4: Advanced Parameters

```bash
curl -X POST "http://localhost:8080/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Advanced synthesis with custom parameters",
    "format": "mp3",
    "normalize": true,
    "chunk_length": 200,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "max_new_tokens": 2048,
    "streaming": false
  }' \
  --output advanced.mp3
```

---

## Python Implementation Example

```python
import requests
import base64
from pathlib import Path


class FishSpeechTTS:
    """
    Client for OpenAudio S1 Mini and Fish Speech 1.5 TTS APIs
    
    Usage:
        # Initialize with model endpoint
        tts = FishSpeechTTS(base_url="http://localhost:8080")
        
        # Basic TTS
        audio = tts.generate("Hello world")
        
        # Voice cloning with reference
        tts.add_reference("my_voice", "reference.wav", "Reference text")
        audio = tts.generate("New text", reference_id="my_voice")
    """
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        
    def generate(
        self,
        text: str,
        reference_id: str = None,
        format: str = "wav",
        normalize: bool = True,
        temperature: float = 0.7,
        output_path: str = None
    ) -> bytes:
        """
        Generate TTS audio from text
        
        Args:
            text: Text to synthesize
            reference_id: Reference voice ID for cloning (optional)
            format: Output format (wav, mp3, flac)
            normalize: Enable text normalization for numbers/dates
            temperature: Creativity (0.1-1.0, default 0.7)
            output_path: Save audio to file (optional)
            
        Returns:
            Audio bytes
        """
        payload = {
            "text": text,
            "format": format,
            "normalize": normalize,
            "temperature": temperature
        }
        
        if reference_id:
            payload["reference_id"] = reference_id
            
        response = requests.post(
            f"{self.base_url}/v1/tts",
            json=payload
        )
        response.raise_for_status()
        
        audio_bytes = response.content
        
        if output_path:
            Path(output_path).write_bytes(audio_bytes)
            
        return audio_bytes
    
    def add_reference(
        self,
        reference_id: str,
        audio_path: str,
        text: str
    ):
        """
        Add reference voice for cloning
        
        Args:
            reference_id: Unique ID for this voice
            audio_path: Path to reference audio file (3-30 seconds)
            text: Exact text spoken in the audio
        """
        with open(audio_path, 'rb') as f:
            files = {
                'id': (None, reference_id),
                'audio': (Path(audio_path).name, f, 'audio/wav'),
                'text': (None, text)
            }
            response = requests.post(
                f"{self.base_url}/v1/references/add",
                files=files
            )
            response.raise_for_status()
            
    def list_references(self) -> list:
        """Get list of all reference voice IDs"""
        response = requests.get(f"{self.base_url}/v1/references/list")
        response.raise_for_status()
        return response.json()
    
    def delete_reference(self, reference_id: str):
        """Delete a reference voice"""
        response = requests.delete(
            f"{self.base_url}/v1/references/delete",
            json={"reference_id": reference_id}
        )
        response.raise_for_status()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

# Example 1: Basic multilingual TTS
tts = FishSpeechTTS("http://localhost:8080")

# English
tts.generate("Hello world", output_path="english.wav")

# Chinese
tts.generate("你好世界", output_path="chinese.wav")

# Korean
tts.generate("안녕하세요", output_path="korean.wav")

# Spanish
tts.generate("Hola mundo", output_path="spanish.wav")

# Example 2: Voice cloning workflow
# Add reference voice
tts.add_reference(
    reference_id="narrator",
    audio_path="narrator_sample.wav",
    text="This is a sample of the narrator's voice"
)

# Generate with cloned voice
tts.generate(
    text="This is new content in the narrator's voice",
    reference_id="narrator",
    output_path="narrator_cloned.wav"
)

# List available voices
voices = tts.list_references()
print(f"Available voices: {voices}")

# Delete reference when done
tts.delete_reference("narrator")

# Example 3: Switching between models
s1_mini = FishSpeechTTS("http://localhost:8080")  # S1 Mini
fish_15 = FishSpeechTTS("http://localhost:8081")  # Fish Speech 1.5

# Compare outputs
s1_audio = s1_mini.generate("Test text", output_path="s1_mini.wav")
fs15_audio = fish_15.generate("Test text", output_path="fish_15.wav")
```

---

## Docker Management Commands

```bash
# View running containers
docker ps

# View logs
docker logs openaudio-s1-mini
docker logs fish-speech-1-5

# Stop containers
docker stop openaudio-s1-mini
docker stop fish-speech-1-5

# Start containers
docker start openaudio-s1-mini
docker start fish-speech-1-5

# Restart containers
docker restart openaudio-s1-mini

# Remove containers
docker rm -f openaudio-s1-mini fish-speech-1-5

# View GPU usage
nvidia-smi
```

---

## API Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text` | **(required)** | Text to synthesize |
| `format` | `wav` | Output format: `wav`, `mp3`, `flac` |
| `normalize` | `true` | Text normalization for numbers/dates |
| `chunk_length` | `200` | Text chunk size for processing |
| `temperature` | `0.7` | Creativity: 0.1 (conservative) to 1.0 (creative) |
| `top_p` | `0.8` | Sampling diversity |
| `repetition_penalty` | `1.1` | Prevent repetition (1.0-2.0) |
| `max_new_tokens` | `1024` | Maximum output length |
| `reference_id` | `null` | Reference voice ID for cloning |
| `references` | `[]` | Inline reference audio array |
| `streaming` | `false` | Enable audio streaming |
| `seed` | `null` | Random seed for reproducibility |
| `use_memory_cache` | `off` | Cache mode: `off`, `on`, `read-only` |

---

## Quick Reference

### Health Check

```bash
curl http://localhost:8080/v1/health  # S1 Mini
curl http://localhost:8081/v1/health  # Fish 1.5
```

### API Documentation

- **OpenAudio S1 Mini**: <http://localhost:8080/>
- **Fish Speech 1.5**: <http://localhost:8081/>

### Model Comparison

| Model | Port | Accuracy | Status |
|-------|------|----------|--------|
| **OpenAudio S1 Mini** | 8080 | 0.8% WER | ✅ Recommended |
| Fish Speech 1.5 | 8081 | 3.5% WER | Legacy |

---

## Notes

- Both models use **identical APIs**, so code is portable between them
- Reference audio should be **3-30 seconds** for best results
- Enable `COMPILE=1` for **~10x faster inference** (requires more VRAM)
- For production, use **Docker Compose** (see `docker-compose.yml`)
