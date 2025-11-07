# OpenAudio S1-mini - Local Development

This folder contains the **OpenAudio S1-mini** source code for local development and testing.

## What's Here

- **Source code** - Full Fish Speech codebase
- **Gradio WebUI** - Interactive web interface (`app.py`)
- **Python API** - Simple programmatic interface (`simple_api.py`)
- **Model code** - TTS inference engine and models

## Quick Start

### WebUI (Interactive)

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/WSL
# or
venv\Scripts\activate  # Windows

# Run WebUI
python app.py
```

Access at: http://localhost:7860

### Python API

```python
from simple_api import OpenAudioTTS

# Initialize (loads models)
tts = OpenAudioTTS()

# Generate audio
audio, sample_rate = tts.generate("Hello world!")

# Save to file
tts.save_audio("output.wav", audio, sample_rate)
```

## For Production

**Use the Docker containers** in the `../docker/` folder instead. They provide:
- ✅ Pre-built official images
- ✅ REST API endpoints
- ✅ Better for production deployment
- ✅ Easier to manage

## Files

- `app.py` - Gradio WebUI
- `simple_api.py` - Python API wrapper
- `requirements.txt` - Python dependencies
- `fish_speech/` - Core TTS engine and models

## Note

This is the development/testing environment. For production API services, use the Docker setup in `../docker/`.
