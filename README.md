# OpenAudio S1-mini Project

Text-to-speech (TTS) project using OpenAudio S1-mini and Fish Speech models with Docker containerization.

## Project Structure

```
.
├── docker/                    # Docker containers for production API
│   ├── docker-compose.yml     # Run both S1 Mini & Fish 1.5 side-by-side
│   ├── test_voice_cloning.py  # Voice cloning test script
│   └── README.md              # Docker setup documentation
│
├── openaudio-s1-mini/        # Source code for local development
│   ├── app.py                 # Gradio WebUI
│   ├── simple_api.py          # Python API wrapper
│   └── fish_speech/           # Core TTS engine
│
└── memory-bank/               # Project documentation
    ├── projectbrief.md        # Project overview
    ├── activeContext.md       # Current work status
    └── progress.md              # Implementation progress
```

## Quick Start

### Production (Docker - Recommended)

```bash
cd docker
docker-compose up -d
```

- **OpenAudio S1 Mini** API: <http://localhost:8080>
- **Fish Speech 1.5** API: <http://localhost:8081>

See `docker/README.md` for details.

### Development (Local)

```bash
cd openaudio-s1-mini
source venv/bin/activate
python app.py
```

WebUI at: <http://localhost:7860>

## What Each Folder Does

### `docker/` - Production API Services

- **Purpose**: Run TTS as REST API services
- **Contains**: Docker Compose config, test scripts
- **Use for**: Production deployment, API integration
- **Models**: OpenAudio S1 Mini (port 8080) + Fish Speech 1.5 (port 8081)

### `openaudio-s1-mini/` - Source Code

- **Purpose**: Local development and testing
- **Contains**: Full source code, Gradio WebUI, Python API
- **Use for**: Development, experimentation, understanding the codebase
- **Note**: Use Docker for production instead

### `memory-bank/` - Documentation

- **Purpose**: Project knowledge base
- **Contains**: Project context, progress tracking, technical notes
- **Use for**: Understanding project history and decisions

## Features

- ✅ **Voice Cloning** - Clone voices from reference audio
- ✅ **Multilingual** - Supports 13+ languages (EN, ZH, JA, KO, ES, FR, DE, AR+)
- ✅ **REST API** - Production-ready HTTP API
- ✅ **Docker** - Containerized deployment
- ✅ **WebUI** - Interactive Gradio interface

## Documentation

- **Docker Setup**: `docker/README.md`
- **Local Development**: `openaudio-s1-mini/README.md`
- **Project Context**: `memory-bank/`

## Quick Test

```bash
# Test voice cloning with multilingual examples
cd docker
python3 test_voice_cloning.py
```

## Model Comparison

| Model | Port | Accuracy | Status |
|-------|------|----------|--------|
| **OpenAudio S1 Mini** | 8080 | 0.8% WER | ✅ Recommended |
| Fish Speech 1.5 | 8081 | 3.5% WER | Legacy |

## License

CC BY-NC-SA 4.0 (see model repository for details)
