# Fish Speech Docker Setup

Docker configuration for running **OpenAudio S1 Mini** and **Fish Speech 1.5** side-by-side using official containers.

> 📖 **For complete setup instructions, examples, and Python SDK, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

## Quick Start

```bash
# 1. Download models (if not done)
cd ~/fish-speech-models
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
hf download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5

# 2. Pull Docker image
docker pull fishaudio/fish-speech:server-cuda

# 3. Start containers
cd docker
docker-compose up -d
```

This starts:
- **OpenAudio S1 Mini** on port **8080** (recommended)
- **Fish Speech 1.5** on port **8081** (legacy/comparison)

## Usage

### Simple TTS

```bash
curl -X POST "http://localhost:8080/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "format": "wav"}' \
  --output output.wav
```

### Voice Cloning with Reference Audio

```python
import requests
import base64

# Read reference audio
with open("reference.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

# Generate with voice cloning
response = requests.post(
    "http://localhost:8080/v1/tts",
    json={
        "text": "This is a voice cloning test",
        "references": [{
            "audio": audio_b64,
            "text": "Reference text"
        }],
        "temperature": 0.9,
        "format": "wav"
    }
)

with open("cloned.wav", "wb") as f:
    f.write(response.content)
```

### Test Script

```bash
# Test voice cloning with multilingual examples
python3 test_voice_cloning.py
```

## Management

```bash
# Start/Stop
docker-compose up -d
docker-compose stop
docker-compose down

# View logs
docker-compose logs -f

# Restart service
docker-compose restart openaudio-s1-mini
```

## API Endpoints

- `POST /v1/tts` - Generate TTS (with optional voice cloning)
- `GET /v1/health` - Health check
- `POST /v1/vqgan/encode` - Encode audio
- `POST /v1/vqgan/decode` - Decode audio
- `POST /v1/chat` - Chat interface

**API Docs:** http://localhost:8080/ (S1 Mini) or http://localhost:8081/ (Fish 1.5)

## Configuration

Update checkpoint path in `docker-compose.yml` if needed:
```yaml
volumes:
  - ~/fish-speech-models/checkpoints:/app/checkpoints
```

## Files

- `docker-compose.yml` - Container configuration
- `test_voice_cloning.py` - Voice cloning test script with multilingual examples
- `SETUP_GUIDE.md` - **Complete setup guide** with all commands, examples, and Python SDK
