# System Patterns

## Architecture Overview

The project follows a dual-deployment pattern:
1. **Local Development**: Full source code with WebUI and Python API
2. **Production Docker**: Official containers with REST API

## Deployment Patterns

### Local Development Pattern
```
User → Gradio WebUI (app.py) → TTSInferenceEngine → Models → Audio Output
User → Python API (simple_api.py) → TTSInferenceEngine → Models → Audio Output
```

### Production Docker Pattern
```
Client → REST API (Docker) → TTSInferenceEngine → Models → Audio Response
```

## Model Loading Pattern

1. **Checkpoint Detection**: Check for existing checkpoints
2. **Download if Missing**: Use `snapshot_download()` or Hugging Face CLI
3. **Path Resolution**: Handle Windows mount permission issues (fallback to home directory)
4. **Model Initialization**: Load Llama model, then decoder
5. **Warmup**: Generate sample to compile model (first generation slow)

## Audio Loading Pattern

1. **Primary**: Try TorchCodec via `torchaudio.load()`
2. **Fallback 1**: Use `librosa.load()` (handles BytesIO and various formats)
3. **Fallback 2**: Detect format from file header, use `soundfile.read()` with temp file
4. **Error Handling**: Graceful degradation with informative errors

## GPU Capability Detection Pattern

1. **Check CUDA Availability**: `torch.cuda.is_available()`
2. **Get GPU Compute Capability**: Query GPU properties
3. **Select Precision**:
   - Compute capability >= 8.0: `bfloat16`
   - Compute capability < 8.0: `float16`
4. **Apply Precision**: Set model precision accordingly

## Text Normalization Pattern

1. **Check Language**: Determine if text is Chinese
2. **Conditional Application**: Only apply normalization if:
   - `normalize=True` AND
   - Text contains Chinese characters
3. **Normalization Process**: Convert numbers, dates, money, etc. to Chinese text
4. **Default Behavior**: Disabled by default (normalize=False)

## Error Handling Patterns

### Permission Errors
- **Windows Mounts**: Fallback to home directory
- **Checkpoints**: `~/openaudio_checkpoints/`
- **Outputs**: `~/openaudio_outputs/`

### Audio Loading Errors
- **TorchCodec Missing**: Fallback to librosa
- **Format Detection**: Auto-detect from file headers
- **BytesIO Handling**: Write to temp file for format detection

### Model Loading Errors
- **Incomplete Downloads**: Detect and re-download
- **Authentication**: Clear error messages with instructions
- **Path Issues**: Automatic path resolution

## Configuration Patterns

### Environment-Based Configuration
- **Local**: Config files and environment variables
- **Docker**: Environment variables in docker-compose.yml
- **Checkpoints**: Volume mounts from host to container

### Default Values
- **Device**: Auto-detect CUDA, fallback to CPU
- **Precision**: Auto-detect based on GPU capability
- **Normalization**: Disabled by default
- **Compilation**: Enabled in Docker (`COMPILE=1`)

## API Patterns

### Local Python API
- **Class-Based**: `OpenAudioTTS` class
- **Method-Based**: `generate()`, `text_to_speech()`, `save_audio()`
- **Error Handling**: Exceptions with clear messages

### Docker REST API
- **RESTful**: Standard HTTP methods (POST, GET, DELETE)
- **JSON Payloads**: Structured request/response
- **File Uploads**: Multipart/form-data for reference audio
- **Base64 Encoding**: For inline audio in JSON

## Testing Patterns

### Test Scripts
- **Single Test File**: `test_voice_cloning.py` for Docker
- **Multilingual Tests**: Test multiple languages in one script
- **Reference Audio**: Test voice cloning with provided audio
- **Output Management**: Save to user-writable directory

## Documentation Patterns

### Hierarchical Documentation
- **Root README**: Overview and folder structure
- **Folder READMEs**: Specific to each folder's purpose
- **Setup Guides**: Step-by-step instructions
- **API Documentation**: Examples and parameter references

### Code Documentation
- **Docstrings**: Clear method/function descriptions
- **Comments**: Explain non-obvious logic
- **Type Hints**: Where applicable

## File Organization Patterns

### Separation of Concerns
- **Source Code**: `openaudio-s1-mini/`
- **Docker Config**: `docker/`
- **Documentation**: `memory-bank/`
- **Root**: Overview and navigation

### Test File Organization
- **Single Test File**: One comprehensive test script
- **Location**: In relevant folder (e.g., `docker/test_voice_cloning.py`)
- **Output**: User-writable directory (home directory)

## Container Communication Patterns

### Docker Compose
- **Service Names**: Used for inter-container communication
- **Network**: Default bridge network
- **Ports**: Exposed to host for external access
- **Volumes**: Shared checkpoint storage

### API Communication
- **HTTP**: Standard REST API
- **JSON**: Request/response format
- **Base64**: Audio encoding for JSON payloads
- **Multipart**: File uploads

