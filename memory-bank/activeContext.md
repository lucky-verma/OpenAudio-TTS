# Active Context

## Current Status

**Application: FULLY OPERATIONAL** ✅

The OpenAudio S1-mini project is fully operational with both local development and production Docker deployment options.

## Recent Achievements

1. **Docker Production Setup Complete**
   - ✅ Official Docker containers configured for OpenAudio S1 Mini and Fish Speech 1.5
   - ✅ Docker Compose setup for running both models side-by-side
   - ✅ Complete setup guide with examples and Python SDK
   - ✅ Voice cloning test script with multilingual examples

2. **Documentation Cleanup**
   - ✅ Consolidated test files (kept single `test_voice_cloning.py`)
   - ✅ Created comprehensive `SETUP_GUIDE.md` with all commands and examples
   - ✅ Updated root README explaining folder structure
   - ✅ Updated `docker/README.md` and `openaudio-s1-mini/README.md`

3. **All Critical Issues Resolved**
   - ✅ TorchCodec error fixed with fallback to librosa/soundfile
   - ✅ bfloat16 warnings eliminated (auto-detects GPU capability, uses float16 for RTX 2070 SUPER)
   - ✅ Reference audio loading working
   - ✅ Text normalization made configurable (disabled by default)
   - ✅ File saving issues on Windows mounts resolved

4. **New Features Added**
   - ✅ Simple Python API (`simple_api.py`) for programmatic access
   - ✅ Example usage scripts
   - ✅ Text normalization checkbox in web UI
   - ✅ Automatic GPU capability detection
   - ✅ Docker Compose for production deployment

## Current Configuration

### Local Development
- **Checkpoints**: `/home/lucki/openaudio_checkpoints/openaudio-s1-mini`
- **Virtual Environment**: `~/openaudio_venv`
- **Device**: CUDA (RTX 2070 SUPER)
- **Precision**: float16 (auto-detected, RTX 2070 SUPER doesn't support bfloat16 natively)
- **Output Directory**: `~/openaudio_outputs/` (fallback for Windows mount issues)
- **WebUI**: http://localhost:7860

### Docker Production
- **OpenAudio S1 Mini**: Port 8080 (recommended)
- **Fish Speech 1.5**: Port 8081 (legacy/comparison)
- **Checkpoints**: `~/fish-speech-models/checkpoints`
- **Docker Image**: `fishaudio/fish-speech:server-cuda`
- **API Docs**: http://localhost:8080/ (S1 Mini), http://localhost:8081/ (Fish 1.5)

## Active Features

- ✅ Basic TTS generation
- ✅ Web interface (Gradio) for local development
- ✅ REST API for production (Docker)
- ✅ Voice cloning with reference audio
- ✅ Multilingual support (English, Spanish, Chinese, Korean, Japanese, etc.)
- ✅ Text normalization (configurable, disabled by default)
- ✅ Simple Python API for local development
- ✅ Python SDK example in SETUP_GUIDE.md

## Project Structure

```
OPENAUDIO_TEST/
├── docker/                      # Production Docker setup
│   ├── docker-compose.yml       # Run both models side-by-side
│   ├── test_voice_cloning.py   # Voice cloning test script
│   ├── README.md                # Quick reference
│   └── SETUP_GUIDE.md          # Complete setup guide with examples
│
├── openaudio-s1-mini/          # Source code for local development
│   ├── app.py                   # Gradio WebUI
│   ├── simple_api.py            # Python API wrapper
│   └── fish_speech/             # Core TTS engine
│
├── memory-bank/                 # Project documentation
│   ├── projectbrief.md
│   ├── activeContext.md
│   ├── progress.md
│   └── techContext.md
│
└── README.md                    # Root - explains folder structure
```

## Known Limitations

1. **Prosody Control**: Emotion and tone markers (e.g., `(excited)`, `(sad)`) may not work as expected. S1-mini is a 0.5B distilled model with limited prosody control compared to the full S1 (4B) model.

2. **Text Normalization**: The normalization system (`ChnNormedText`) is designed for Chinese text and converts numbers to Chinese numerals. It should be:
   - **Enabled** for Chinese text
   - **Disabled** for English, Korean, Japanese, and other languages

3. **First Generation Speed**: First generation is slow (~2-3 minutes) due to PyTorch compilation. Subsequent generations are much faster.

## Recent Code Changes

### Docker Setup
- Created `docker/docker-compose.yml` for running both models
- Created `docker/SETUP_GUIDE.md` with comprehensive instructions
- Created `docker/test_voice_cloning.py` for testing voice cloning
- Updated `docker/README.md` with quick reference

### Documentation
- Created root `README.md` explaining folder structure
- Updated `openaudio-s1-mini/README.md` for local development
- Cleaned up test files (removed duplicates, kept single test file)

### Key Files Modified (Previous)
- `app.py`: Added GPU capability detection, text normalization checkbox
- `fish_speech/inference_engine/reference_loader.py`: Added TorchCodec fallback, librosa/soundfile support
- `tools/webui/inference.py`: Added normalize parameter
- `simple_api.py`: Created for programmatic access

## Current Focus

The project is production-ready with two deployment options:

1. **Local Development** (`openaudio-s1-mini/`):
   - WebUI for interactive testing
   - Python API for programmatic access
   - Full source code access

2. **Production Docker** (`docker/`):
   - Official containers for reliability
   - REST API endpoints
   - Run both models simultaneously
   - Complete setup guide with examples

## Next Steps (Optional Enhancements)

- Monitor Docker container performance
- Add batch processing utilities
- Consider language-aware text normalization
- Optimize first-generation compilation time
