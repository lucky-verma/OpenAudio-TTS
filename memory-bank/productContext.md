# Product Context

## Why This Project Exists

This project provides a complete text-to-speech (TTS) solution using OpenAudio S1-mini and Fish Speech models, offering both local development and production deployment options. It enables users to generate high-quality, multilingual speech synthesis with voice cloning capabilities.

## Problems It Solves

1. **High-Quality TTS**: Provides state-of-the-art TTS with 0.8% WER (OpenAudio S1 Mini)
2. **Voice Cloning**: Enables zero-shot and few-shot voice cloning from reference audio
3. **Multilingual Support**: Supports 13+ languages (EN, ZH, JA, KO, ES, FR, DE, AR+)
4. **Flexible Deployment**: Both local development (WebUI) and production (Docker REST API)
5. **Easy Integration**: Simple Python API and REST API for easy integration
6. **Model Comparison**: Run both OpenAudio S1 Mini and Fish Speech 1.5 side-by-side

## How It Should Work

### Local Development Workflow
1. User activates virtual environment
2. User runs `python app.py` to start WebUI
3. User enters text in WebUI or uses Python API
4. System generates audio with optional voice cloning
5. User downloads or saves generated audio

### Production Docker Workflow
1. User downloads model weights
2. User starts Docker containers with `docker-compose up -d`
3. Client applications send HTTP requests to REST API
4. API generates audio and returns audio file
5. Client applications use audio for their purposes

### Voice Cloning Workflow
1. User provides reference audio (3-30 seconds)
2. User provides reference text (what's spoken in audio)
3. System extracts voice characteristics
4. User generates new text with cloned voice
5. System synthesizes audio in cloned voice

## User Experience Goals

### For Developers
- **Easy Setup**: Clear documentation and setup guides
- **Flexible API**: Both Python API and REST API
- **Full Control**: Access to all model parameters
- **Good Examples**: Comprehensive examples and test scripts

### For End Users (via WebUI)
- **Simple Interface**: Intuitive Gradio WebUI
- **Quick Generation**: Fast audio generation after warmup
- **Voice Cloning**: Easy reference audio upload
- **Multilingual**: Support for multiple languages

### For Production Deployments
- **Reliable**: Official Docker containers
- **Scalable**: REST API for multiple clients
- **Documented**: Complete API documentation
- **Tested**: Test scripts for validation

## Key Features

1. **Text-to-Speech Generation**
   - High-quality speech synthesis
   - Multilingual support
   - Configurable parameters (temperature, top_p, etc.)

2. **Voice Cloning**
   - Zero-shot cloning from reference audio
   - Few-shot cloning with multiple references
   - Persistent reference voices (Docker API)

3. **Text Normalization**
   - Chinese text normalization (numbers, dates, money)
   - Configurable per request
   - Disabled by default

4. **Multiple Deployment Options**
   - Local development with WebUI
   - Production Docker with REST API
   - Python API for programmatic access

5. **Model Comparison**
   - Run both models simultaneously
   - Compare quality and performance
   - Choose best model for use case

## User Personas

### Developer/Researcher
- **Needs**: Full control, experimentation, understanding
- **Uses**: Local development, source code access
- **Tools**: WebUI, Python API, source code

### Application Developer
- **Needs**: Integration, reliability, scalability
- **Uses**: Docker REST API
- **Tools**: REST API, Docker containers

### Content Creator
- **Needs**: Voice cloning, multilingual support
- **Uses**: WebUI or API
- **Tools**: WebUI for easy use, API for automation

## Success Metrics

1. **Quality**: 0.8% WER (OpenAudio S1 Mini)
2. **Speed**: ~70-80 tokens/sec after warmup
3. **Languages**: 13+ languages supported
4. **Voice Cloning**: Works with 3-30 second reference audio
5. **Deployment**: Both local and Docker working
6. **Documentation**: Complete setup guides and examples

## Use Cases

1. **Content Creation**: Generate voiceovers in multiple languages
2. **Accessibility**: Text-to-speech for accessibility tools
3. **Localization**: Generate speech in multiple languages
4. **Voice Cloning**: Clone voices for specific applications
5. **Research**: Experiment with TTS models
6. **Production Services**: Deploy TTS as a service

## Limitations & Considerations

1. **Prosody Control**: Limited in S1-mini model (0.5B)
2. **Text Normalization**: Only for Chinese text
3. **First Generation**: Slow due to compilation
4. **GPU Requirements**: 12GB+ VRAM recommended for Docker with compilation
5. **License**: CC-BY-NC-SA-4.0 (non-commercial use)

## Future Enhancements (Potential)

1. Language-aware text normalization
2. Improved prosody control
3. Batch processing utilities
4. Streaming audio generation
5. Additional language support

