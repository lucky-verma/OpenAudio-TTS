"""
FastAPI service for OpenAudio S1-mini TTS API

This service exposes all TTS functionality via REST API endpoints.
"""

import os
import sys
import base64
import io
from pathlib import Path
from typing import Optional, List
import tempfile

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from loguru import logger

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="OpenAudio S1-mini TTS API",
    description="REST API for OpenAudio S1-mini text-to-speech model",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
inference_engine: Optional[TTSInferenceEngine] = None
decoder_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
sample_rate = 44100  # Model default


def get_precision():
    """Auto-detect precision based on GPU capability."""
    if device == "cuda" and torch.cuda.is_available():
        try:
            compute_capability = torch.cuda.get_device_capability()
            # Ampere (8.0+) supports bfloat16 natively
            if compute_capability[0] >= 8:
                return torch.bfloat16
            else:
                # Older GPUs (Turing, Pascal, etc.) - use float16
                return torch.half
        except Exception:
            return torch.half
    else:
        # CPU - use bfloat16
        return torch.bfloat16


def load_models():
    """Load TTS models on startup."""
    global inference_engine, decoder_model, device, sample_rate
    
    logger.info("Loading TTS models...")
    
    # Get checkpoint path from environment or default
    checkpoint_path = os.getenv(
        "CHECKPOINT_PATH",
        "/app/checkpoints/openaudio-s1-mini"
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}\n"
            f"Please set CHECKPOINT_PATH environment variable or mount checkpoints volume."
        )
    
    # Auto-detect device
    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    precision = get_precision()
    compile_model = os.getenv("COMPILE", "true").lower() == "true"
    
    logger.info(f"Device: {device}, Precision: {precision}, Compile: {compile_model}")
    
    # Load LLAMA model
    logger.info("Loading LLAMA model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=checkpoint_path,
        device=device,
        precision=precision,
        compile=compile_model,
    )
    
    # Load decoder model
    logger.info("Loading decoder model...")
    decoder_model = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path=f"{checkpoint_path}/codec.pth",
        device=device,
    )
    
    # Get sample rate from model
    if hasattr(decoder_model, "spec_transform"):
        sample_rate = decoder_model.spec_transform.sample_rate
    else:
        sample_rate = decoder_model.sample_rate
    
    # Create inference engine
    logger.info("Creating inference engine...")
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=compile_model,
        precision=precision,
    )
    
    # Warmup
    logger.info("Warming up model...")
    try:
        warmup_request = ServeTTSRequest(
            text="Hello world.",
            references=[],
            max_new_tokens=1024,
            chunk_length=0,
            top_p=0.9,
            repetition_penalty=1.1,
            temperature=0.9,
            format="wav",
        )
        list(inference_engine.inference(warmup_request))
        logger.info("✓ Model warmed up")
    except Exception as e:
        logger.warning(f"Warmup failed (this is usually okay): {e}")
    
    logger.info("✓ All models loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


# Request/Response models
class TTSGenerateJSONRequest(BaseModel):
    """JSON request model for TTS generation."""
    text: str = Field(..., description="Text to convert to speech")
    reference_audio_base64: Optional[str] = Field(None, description="Base64 encoded reference audio")
    reference_text: Optional[str] = Field(None, description="Transcription of reference audio")
    max_new_tokens: int = Field(1024, ge=0, description="Maximum tokens to generate (0 = no limit)")
    chunk_length: int = Field(0, ge=0, description="Iterative prompt length (0 = off, 100-300 when enabled)")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p sampling")
    repetition_penalty: float = Field(1.1, ge=0.9, le=2.0, description="Repetition penalty")
    temperature: float = Field(0.9, ge=0.1, le=1.0, description="Sampling temperature")
    seed: Optional[int] = Field(None, description="Random seed (None = randomized)")
    normalize: bool = Field(False, description="Enable text normalization (for Chinese)")
    use_memory_cache: str = Field("off", pattern="^(on|off)$", description="Use memory cache")
    output_format: str = Field("wav", pattern="^(wav|mp3|pcm)$", description="Output format")
    
    @field_validator("chunk_length")
    @classmethod
    def validate_chunk_length(cls, v):
        """Validate chunk_length: must be 0 or 100-300."""
        if v == 0:
            return 0
        elif 100 <= v <= 300:
            return v
        else:
            raise ValueError("chunk_length must be 0 (disabled) or between 100-300")
    
    @field_validator("chunk_length", mode="before")
    @classmethod
    def coerce_chunk_length(cls, v):
        """Coerce chunk_length to proper type."""
        if isinstance(v, str):
            v = int(v)
        return v


class TTSGenerateJSONResponse(BaseModel):
    """JSON response model for TTS generation."""
    audio_base64: str = Field(..., description="Base64 encoded audio file")
    sample_rate: int = Field(44100, description="Sample rate in Hz")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    tokens_generated: Optional[int] = Field(None, description="Number of tokens generated")
    format: str = Field(..., description="Audio format")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    gpu_memory_mb: Optional[int] = None


class InfoResponse(BaseModel):
    """API info response."""
    version: str
    model: str
    sample_rate: int
    parameters: dict


def validate_chunk_length(value: int) -> int:
    """Validate chunk_length parameter."""
    if value == 0:
        return 0
    elif 100 <= value <= 300:
        return value
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"chunk_length must be 0 (disabled) or between 100-300, got {value}"
        )


def convert_audio_to_bytes(audio: np.ndarray, sample_rate: int, format: str) -> bytes:
    """Convert audio array to bytes in specified format."""
    buffer = io.BytesIO()
    
    if format == "wav":
        sf.write(buffer, audio, sample_rate, format="WAV")
    elif format == "mp3":
        # MP3 requires ffmpeg
        sf.write(buffer, audio, sample_rate, format="MP3")
    elif format == "pcm":
        # Raw PCM float16
        audio_float16 = audio.astype(np.float16)
        buffer.write(audio_float16.tobytes())
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return buffer.getvalue()


def get_content_type(format: str) -> str:
    """Get Content-Type for audio format."""
    if format == "wav":
        return "audio/wav"
    elif format == "mp3":
        return "audio/mpeg"
    elif format == "pcm":
        return "application/octet-stream"
    else:
        return "application/octet-stream"


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_memory = None
    if device == "cuda" and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    
    return HealthResponse(
        status="healthy" if inference_engine is not None else "loading",
        model_loaded=inference_engine is not None,
        device=device,
        gpu_memory_mb=gpu_memory,
    )


@app.get("/api/v1/info", response_model=InfoResponse)
async def get_info():
    """Get API information and parameter ranges."""
    return InfoResponse(
        version="1.0.0",
        model="openaudio-s1-mini",
        sample_rate=sample_rate,
        parameters={
            "max_new_tokens": {"min": 0, "max": None, "default": 1024},
            "chunk_length": {"min": 0, "max": 300, "default": 0, "note": "Must be 0 or 100-300 when enabled"},
            "top_p": {"min": 0.1, "max": 1.0, "default": 0.9},
            "repetition_penalty": {"min": 0.9, "max": 2.0, "default": 1.1},
            "temperature": {"min": 0.1, "max": 1.0, "default": 0.9},
        }
    )


@app.post("/api/v1/tts/generate")
async def generate_tts_file(
    text: str = Form(..., description="Text to convert to speech"),
    reference_audio: Optional[UploadFile] = File(None, description="Reference audio file for voice cloning"),
    reference_text: Optional[str] = Form(None, description="Transcription of reference audio"),
    reference_id: Optional[str] = Form(None, description="Reference ID (alternative to reference_audio)"),
    max_new_tokens: int = Form(1024, description="Maximum tokens to generate (0 = no limit)"),
    chunk_length: int = Form(0, description="Iterative prompt length (0 = off, 100-300 when enabled)"),
    top_p: float = Form(0.9, description="Top-p sampling (0.1-1.0)"),
    repetition_penalty: float = Form(1.1, description="Repetition penalty (0.9-2.0)"),
    temperature: float = Form(0.9, description="Sampling temperature (0.1-1.0)"),
    seed: Optional[int] = Form(None, description="Random seed (None = randomized)"),
    normalize: bool = Form(False, description="Enable text normalization (for Chinese)"),
    use_memory_cache: str = Form("off", description="Use memory cache (on/off)"),
    output_format: str = Form("wav", description="Output format (wav/mp3/pcm)"),
):
    """
    Generate TTS audio from text (file upload endpoint).
    
    Returns audio file directly.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet"
        )
    
    # Validate parameters
    try:
        chunk_length = validate_chunk_length(chunk_length)
    except HTTPException:
        raise
    
    # Validate text length
    max_text_length = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    if len(text) > max_text_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text too long. Maximum length is {max_text_length} characters."
        )
    
    # Validate output format
    if output_format not in ["wav", "mp3", "pcm"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid output_format. Must be 'wav', 'mp3', or 'pcm'"
        )
    
    # Prepare references
    references = []
    if reference_audio:
        # Read uploaded file
        max_size_mb = int(os.getenv("MAX_AUDIO_SIZE_MB", "10"))
        audio_bytes = await reference_audio.read()
        if len(audio_bytes) > max_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Reference audio too large. Maximum size is {max_size_mb}MB."
            )
        references.append(
            ServeReferenceAudio(
                audio=audio_bytes,
                text=reference_text or "",
            )
        )
    
    # Create request
    # Note: Schema requires chunk_length to be 100-300, but 0 means disabled
    # When chunk_length is 0, we use 200 (default) for schema validation,
    # but the inference engine checks chunk_length > 0 to determine iterative_prompt
    try:
        # Use 200 (schema default) when chunk_length is 0, but inference engine will see it as disabled
        schema_chunk_length = chunk_length if chunk_length > 0 else 200
        
        request = ServeTTSRequest(
            text=text,
            references=references,
            reference_id=reference_id,
            max_new_tokens=max_new_tokens,
            chunk_length=schema_chunk_length,  # Schema requires 100-300, use 200 when disabled
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            normalize=normalize,
            use_memory_cache=use_memory_cache,
            format=output_format,
        )
        
        # Override chunk_length in the request dict if it was 0
        # The inference engine uses iterative_prompt=req.chunk_length > 0
        # So we need to ensure chunk_length is 0 in the actual request
        if chunk_length == 0:
            # We'll need to modify the request after creation
            # Actually, the inference engine reads from req.chunk_length directly
            # So we need to patch it
            request.chunk_length = 0
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )
    
    # Generate audio
    try:
        results = list(inference_engine.inference(request))
        
        # Extract final audio
        audio_array = None
        for result in results:
            if result.code == "final" and result.audio is not None:
                result_sample_rate, audio_array = result.audio
                break
            elif result.code == "error" and result.error is not None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Generation error: {str(result.error)}"
                )
        
        if audio_array is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No audio generated"
            )
        
        # Convert to bytes
        audio_bytes = convert_audio_to_bytes(audio_array, sample_rate, output_format)
        
        # Return audio file
        return Response(
            content=audio_bytes,
            media_type=get_content_type(output_format),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{output_format}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio generation failed: {str(e)}"
        )


@app.post("/api/v1/tts/generate-json", response_model=TTSGenerateJSONResponse)
async def generate_tts_json(request: TTSGenerateJSONRequest):
    """
    Generate TTS audio from text (JSON endpoint with base64 audio).
    
    Returns JSON with base64 encoded audio.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet"
        )
    
    # Validate text length
    max_text_length = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    if len(request.text) > max_text_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text too long. Maximum length is {max_text_length} characters."
        )
    
    # Validate chunk_length
    try:
        chunk_length = validate_chunk_length(request.chunk_length)
    except HTTPException:
        raise
    
    # Prepare references
    references = []
    if request.reference_audio_base64:
        try:
            audio_bytes = base64.b64decode(request.reference_audio_base64)
            max_size_mb = int(os.getenv("MAX_AUDIO_SIZE_MB", "10"))
            if len(audio_bytes) > max_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Reference audio too large. Maximum size is {max_size_mb}MB."
                )
            references.append(
                ServeReferenceAudio(
                    audio=audio_bytes,
                    text=request.reference_text or "",
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 audio: {str(e)}"
            )
    
    # Create request
    # Handle chunk_length=0 (schema requires 100-300, but 0 means disabled)
    schema_chunk_length = chunk_length if chunk_length > 0 else 200
    
    try:
        tts_request = ServeTTSRequest(
            text=request.text,
            references=references,
            reference_id=None,
            max_new_tokens=request.max_new_tokens,
            chunk_length=schema_chunk_length,  # Use 200 when disabled for schema validation
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            seed=request.seed,
            normalize=request.normalize,
            use_memory_cache=request.use_memory_cache,
            format=request.output_format,
        )
        
        # Override chunk_length if it was 0 (disabled)
        if chunk_length == 0:
            object.__setattr__(tts_request, 'chunk_length', 0)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )
    
    # Generate audio
    try:
        results = list(inference_engine.inference(tts_request))
        
        # Extract final audio
        audio_array = None
        tokens_generated = None
        for result in results:
            if result.code == "final" and result.audio is not None:
                result_sample_rate, audio_array = result.audio
                # Try to get token count from result if available
                if hasattr(result, "tokens_generated"):
                    tokens_generated = result.tokens_generated
                break
            elif result.code == "error" and result.error is not None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Generation error: {str(result.error)}"
                )
        
        if audio_array is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No audio generated"
            )
        
        # Convert to bytes and encode
        audio_bytes = convert_audio_to_bytes(audio_array, sample_rate, request.output_format)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Calculate duration
        duration = len(audio_array) / sample_rate
        
        return TTSGenerateJSONResponse(
            audio_base64=audio_base64,
            sample_rate=sample_rate,
            duration_seconds=duration,
            tokens_generated=tokens_generated,
            format=request.output_format,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )

