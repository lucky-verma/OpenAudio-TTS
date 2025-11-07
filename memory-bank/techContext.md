# Technical Context

## Technology Stack

### Core Technologies
- **Python 3.12**: Primary programming language (WSL environment)
- **PyTorch**: Deep learning framework
- **Gradio**: Web interface framework (local development)
- **FastAPI/Kui**: REST API framework (Docker production)
- **Hugging Face Hub**: Model hosting and download
- **TorchCodec**: Audio codec for reference audio loading
- **Docker**: Containerization for production deployment

### Key Dependencies
- `torch` / `torchaudio`: PyTorch and audio processing
- `torchcodec`: Audio codec (required for reference audio)
- `gradio`: Web UI framework (local development)
- `huggingface_hub`: Model download
- `librosa` / `soundfile`: Audio processing (fallback for TorchCodec)
- `transformers`: NLP models
- `faster-whisper`: Speech recognition
- `funasr`: ASR toolkit
- `requests`: HTTP client (for Docker API)

### Model Architecture
- **OpenAudio S1-mini**: 0.5B parameter TTS model
- **Fish Speech 1.5**: 500M parameter TTS model (legacy)
- Based on DAC (Descript Audio Codec) and Qwen3
- Supports zero-shot and few-shot voice cloning
- Multilingual support (English, Chinese, Japanese, Korean, Spanish, etc.)
- Limited prosody control (emotions/tones) compared to full S1 model

## System Requirements

### Hardware
- **GPU**: NVIDIA GeForce RTX 2070 SUPER (8GB VRAM)
  - Compute capability: 7.5 (Turing architecture)
  - Doesn't support bfloat16 natively (uses float16 instead)
- **Memory**: ~5-6 GB GPU memory (basic), ~11-12 GB with reference audio
- **Storage**: ~5GB for model weights

### Software
- Python 3.12
- CUDA (for GPU acceleration)
- WSL (Windows Subsystem for Linux) - for local development
- Docker (for production deployment)
- Git (for cloning repository)

## Project Structure

```
OPENAUDIO_TEST/
├── docker/                      # Production Docker setup
│   ├── docker-compose.yml       # Run both models side-by-side
│   ├── test_voice_cloning.py    # Voice cloning test script
│   ├── README.md                # Quick reference
│   └── SETUP_GUIDE.md          # Complete setup guide with examples
│
├── openaudio-s1-mini/          # Source code for local development
│   ├── app.py                   # Main Gradio application
│   ├── simple_api.py            # Simple Python API wrapper
│   ├── requirements.txt         # Python dependencies
│   ├── checkpoints/            # Model weights (auto-downloaded)
│   │   └── openaudio-s1-mini/  # Actual checkpoint location
│   ├── samples/                # Audio samples
│   │   ├── reference/          # Voice cloning references
│   │   └── outputs/           # Generated samples
│   ├── fish_speech/            # Core model code
│   │   ├── inference_engine/   # TTS inference engine
│   │   │   ├── __init__.py     # Main TTSInferenceEngine
│   │   │   └── reference_loader.py  # Reference audio loading
│   │   ├── models/             # Model implementations
│   │   │   ├── text2semantic/ # LLAMA text-to-semantic model
│   │   │   └── dac/            # DAC audio decoder
│   │   └── text/               # Text processing
│   │       └── chn_text_norm/  # Chinese text normalization
│   └── tools/                  # Fish-speech tools
│       ├── webui/              # WebUI components
│       │   └── inference.py    # Inference wrapper
│       └── api.py              # HTTP API
│
├── memory-bank/                # Project documentation
│   ├── projectbrief.md
│   ├── activeContext.md
│   ├── progress.md
│   └── techContext.md
│
└── README.md                   # Root - explains folder structure
```

## Configuration

### Model Paths

#### Local Development
- Default checkpoint: `checkpoints/openaudio-s1-mini` (falls back to `~/openaudio_checkpoints/openaudio-s1-mini` on Windows mounts)
- Decoder checkpoint: `checkpoints/openaudio-s1-mini/codec.pth`
- Config: `modded_dac_vq`

#### Docker Production
- Checkpoints: `~/fish-speech-models/checkpoints`
- OpenAudio S1 Mini: `checkpoints/openaudio-s1-mini`
- Fish Speech 1.5: `checkpoints/fish-speech-1.5`
- Mounted to: `/app/checkpoints` in container

### Device Selection
- Auto-detects CUDA availability
- Falls back to CPU if GPU unavailable
- Can be manually specified: `--device cuda` or `--device cpu`

### Precision Selection
- **Auto-detection**: Checks GPU compute capability
  - Compute capability >= 8.0 (Ampere+): Uses `bfloat16`
  - Compute capability < 8.0 (Turing/Pascal): Uses `float16`
  - RTX 2070 SUPER: Uses `float16` (compute capability 7.5)
- Manual override: `--half` flag forces `float16`

### Inference Settings
- Default precision: Auto-detected (float16 for RTX 2070 SUPER)
- Compilation: Enabled by default for faster inference (`COMPILE=1` in Docker)
- Max tokens: Configurable via UI or API (default: 1024)
- Text normalization: Configurable via UI (default: False)

### Audio Loading
- Primary: TorchCodec (if installed)
- Fallback: librosa (handles BytesIO and various formats)
- Last resort: soundfile with temp file and format detection

## Development Environment

### Local Development

#### Virtual Environment
- Location: `~/openaudio_venv` (WSL home directory, avoids Windows mount issues)
- Activation: `source ~/openaudio_venv/bin/activate`
- Path stored in: `.venv_path` file

#### Installation
All dependencies installed via `requirements.txt`:
```bash
pip install -r requirements.txt
pip install torchcodec  # Required for reference audio
```

#### WSL-Specific Considerations
- Virtual environment created in home directory (`~/openaudio_venv`) to avoid Windows mount permission issues
- Checkpoints stored in `~/openaudio_checkpoints/` if permission denied in project directory
- Output files saved to `~/openaudio_outputs/` if permission denied on Windows mount
- Uses `python3` instead of `python`

### Docker Production

#### Docker Image
- **Image**: `fishaudio/fish-speech:server-cuda`
- **Base**: PyTorch with CUDA 11.8
- **Ports**: 8080 (S1 Mini), 8081 (Fish 1.5)

#### Docker Compose
- Configuration: `docker/docker-compose.yml`
- Services: `openaudio-s1-mini` and `fish-speech-1-5`
- Volume mounts: Checkpoints from host to container
- Environment variables: `COMPILE=1`, checkpoint paths, decoder config

#### API Endpoints
- `POST /v1/tts` - Generate TTS (with optional voice cloning)
- `GET /v1/health` - Health check
- `POST /v1/vqgan/encode` - Encode audio
- `POST /v1/vqgan/decode` - Decode audio
- `POST /v1/chat` - Chat interface
- `POST /v1/references/add` - Add reference voice
- `GET /v1/references/list` - List reference voices
- `DELETE /v1/references/delete` - Delete reference voice

## Model Access

### Hugging Face
- Model repository: `fishaudio/openaudio-s1-mini` (S1 Mini)
- Model repository: `fishaudio/fish-speech-1.5` (Fish Speech 1.5)
- License: CC-BY-NC-SA-4.0 (non-commercial)
- Requires accepting terms on Hugging Face website
- Requires authentication: `huggingface-cli login` or `hf auth login`

### Download
Model weights download automatically on first run via `snapshot_download()` in `app.py` (local) or via Hugging Face CLI for Docker:
```bash
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
hf download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

## Key Files

### Main Application (Local)
- `app.py`: Main application with Gradio interface
  - Handles checkpoint download
  - GPU capability detection
  - Model loading
  - Web UI setup

### Core TTS Engine
- `fish_speech/inference_engine/__init__.py`: Main TTSInferenceEngine class
- `fish_speech/inference_engine/reference_loader.py`: Reference audio loading with fallbacks
- `fish_speech/inference_engine/vq_manager.py`: VQ token management

### Model Implementations
- `fish_speech/models/text2semantic/inference.py`: LLAMA text-to-semantic model
- `fish_speech/models/dac/inference.py`: DAC audio decoder

### Text Processing
- `fish_speech/text/chn_text_norm/text.py`: Chinese text normalization
  - Converts numbers to Chinese numerals
  - Normalizes dates, money, phone numbers, percentages, etc.
  - Should only be used for Chinese text

### WebUI (Local)
- `tools/webui/inference.py`: WebUI inference wrapper
  - Handles Gradio interface
  - Manages text normalization parameter

### API (Local)
- `simple_api.py`: Simple Python API wrapper
  - `OpenAudioTTS` class for programmatic access
  - Methods: `generate()`, `text_to_speech()`, `save_audio()`

### Docker Production
- `docker/docker-compose.yml`: Docker Compose configuration
- `docker/test_voice_cloning.py`: Voice cloning test script
- `docker/SETUP_GUIDE.md`: Complete setup guide with Python SDK example

## Text Normalization

### Purpose
Normalizes Chinese text for better TTS stability, especially for numbers.

### What It Does
- Converts numbers to Chinese numerals (e.g., "3-1" → "三-一")
- Normalizes dates, money, phone numbers, percentages
- Designed specifically for Chinese text

### When to Use
- **Enable**: For Chinese text
- **Disable**: For English, Korean, Japanese, and other languages

### Implementation
- Controlled via `normalize` parameter in `ServeTTSRequest`
- Default: `False` (disabled)
- Web UI: Checkbox in "Advanced Config" tab
- API: `normalize` parameter in request

## Known Technical Limitations

1. **Prosody Control**: Limited in S1-mini (0.5B model). Emotion/tone markers may not work as expected.

2. **First Generation Speed**: Slow due to PyTorch compilation (~2-3 minutes). Subsequent generations are much faster.

3. **GPU Memory**: Can use up to ~12 GB with reference audio on RTX 2070 SUPER.

4. **Text Normalization**: Only designed for Chinese. Should be disabled for other languages.

5. **Docker**: Requires NVIDIA GPU with 12GB+ VRAM for optimal performance with compilation enabled.
