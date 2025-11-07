import io
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, Tuple

import torch
import torchaudio
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC
from fish_speech.utils.file import (
    AUDIO_EXTENSIONS,
    audio_to_bytes,
    list_files,
    read_ref_text,
)
from fish_speech.utils.schema import ServeReferenceAudio


class ReferenceLoader:

    def __init__(self) -> None:
        """
        Component of the TTSInferenceEngine class.
        Loads and manages the cache for the reference audio and text.
        """
        self.ref_by_id: dict = {}
        self.ref_by_hash: dict = {}

        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.encode_reference: Callable

        # Define the torchaudio backend
        # Handle different torchaudio versions
        try:
            backends = torchaudio.list_audio_backends()
            if "ffmpeg" in backends:
                self.backend = "ffmpeg"
            else:
                self.backend = "soundfile"
        except AttributeError:
            # list_audio_backends() doesn't exist in this version
            # Default to soundfile which is the most common backend
            self.backend = "soundfile"

    def load_by_id(
        self,
        id: str,
        use_cache: Literal["on", "off"],
    ) -> Tuple:

        # Load the references audio and text by id
        ref_folder = Path("references") / id
        ref_folder.mkdir(parents=True, exist_ok=True)
        ref_audios = list_files(
            ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=False
        )

        if use_cache == "off" or id not in self.ref_by_id:
            # If the references are not already loaded, encode them
            prompt_tokens = [
                self.encode_reference(
                    # decoder_model=self.decoder_model,
                    reference_audio=audio_to_bytes(str(ref_audio)),
                    enable_reference_audio=True,
                )
                for ref_audio in ref_audios
            ]
            prompt_texts = [
                read_ref_text(str(ref_audio.with_suffix(".lab")))
                for ref_audio in ref_audios
            ]
            self.ref_by_id[id] = (prompt_tokens, prompt_texts)

        else:
            # Reuse already encoded references
            logger.info("Use same references")
            prompt_tokens, prompt_texts = self.ref_by_id[id]

        return prompt_tokens, prompt_texts

    def load_by_hash(
        self,
        references: list[ServeReferenceAudio],
        use_cache: Literal["on", "off"],
    ) -> Tuple:

        # Load the references audio and text by hash
        audio_hashes = [sha256(ref.audio).hexdigest() for ref in references]

        cache_used = False
        prompt_tokens, prompt_texts = [], []
        for i, ref in enumerate(references):
            if use_cache == "off" or audio_hashes[i] not in self.ref_by_hash:
                # If the references are not already loaded, encode them
                prompt_tokens.append(
                    self.encode_reference(
                        reference_audio=ref.audio,
                        enable_reference_audio=True,
                    )
                )
                prompt_texts.append(ref.text)
                self.ref_by_hash[audio_hashes[i]] = (prompt_tokens[-1], ref.text)

            else:
                # Reuse already encoded references
                cached_token, cached_text = self.ref_by_hash[audio_hashes[i]]
                prompt_tokens.append(cached_token)
                prompt_texts.append(cached_text)
                cache_used = True

        if cache_used:
            logger.info("Use same references")

        return prompt_tokens, prompt_texts

    def load_audio(self, reference_audio, sr):
        """
        Load the audio data from a file or bytes.
        """
        is_bytes = False
        if len(reference_audio) > 255 or not Path(reference_audio).exists():
            audio_data = reference_audio
            reference_audio = io.BytesIO(audio_data)
            is_bytes = True

        # Try to load with torchaudio, with fallback to librosa/soundfile
        try:
            waveform, original_sr = torchaudio.load(reference_audio, backend=self.backend)
        except (ImportError, RuntimeError) as e:
            # If torchcodec is not available or other error, try librosa with temp file
            if "torchcodec" in str(e).lower() or "TorchCodec" in str(e):
                logger.warning(f"TorchCodec not available, falling back to librosa: {e}")
                try:
                    import librosa
                    import tempfile
                    import os
                    
                    # Librosa needs a file path to auto-detect format, not BytesIO
                    # So we write to temp file first
                    if is_bytes:
                        reference_audio.seek(0)
                        audio_bytes = reference_audio.read()
                        # Write to temp file without extension - librosa will auto-detect
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        try:
                            # Librosa can auto-detect format from file path
                            data, original_sr = librosa.load(tmp_path, sr=None, mono=False)
                            # Convert to torch tensor
                            if data.ndim == 1:
                                waveform = torch.from_numpy(data).unsqueeze(0)
                            else:
                                waveform = torch.from_numpy(data)
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                    else:
                        # File path - librosa can handle it directly
                        data, original_sr = librosa.load(reference_audio, sr=None, mono=False)
                        if data.ndim == 1:
                            waveform = torch.from_numpy(data).unsqueeze(0)
                        else:
                            waveform = torch.from_numpy(data)
                except Exception as librosa_error:
                    # Last resort: try soundfile with temp file and format detection
                    logger.warning(f"Librosa failed, trying soundfile with format detection: {librosa_error}")
                    try:
                        import soundfile as sf
                        import tempfile
                        import os
                        
                        if is_bytes:
                            reference_audio.seek(0)
                            audio_bytes = reference_audio.read()
                            # Try to detect format from file header
                            format_detected = False
                            
                            # Check for WAV (RIFF header)
                            if audio_bytes[:4] == b'RIFF' and b'WAVE' in audio_bytes[:12]:
                                suffix = '.wav'
                            # Check for FLAC
                            elif audio_bytes[:4] == b'fLaC':
                                suffix = '.flac'
                            # Check for OGG
                            elif audio_bytes[:4] == b'OggS':
                                suffix = '.ogg'
                            # Check for MP3 (ID3 or frame sync)
                            elif audio_bytes[:3] == b'ID3' or (audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
                                suffix = '.mp3'
                            # Check for M4A (ftyp box)
                            elif b'ftyp' in audio_bytes[:20] or b'mp4' in audio_bytes[:20].lower():
                                suffix = '.m4a'
                            else:
                                # Default to WAV and try others
                                suffix = '.wav'
                            
                            # Try detected format first, then others
                            formats_to_try = [suffix] + [f for f in ['.wav', '.mp3', '.flac', '.ogg', '.m4a'] if f != suffix]
                            
                            for fmt in formats_to_try:
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=fmt) as tmp_file:
                                        tmp_file.write(audio_bytes)
                                        tmp_path = tmp_file.name
                                    try:
                                        data, original_sr = sf.read(tmp_path)
                                        waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data)
                                        format_detected = True
                                        break  # Success
                                    except Exception:
                                        if os.path.exists(tmp_path):
                                            os.unlink(tmp_path)
                                        continue
                                except Exception:
                                    continue
                            
                            if not format_detected:
                                raise Exception("Could not determine audio format from bytes")
                        else:
                            # File path - soundfile can handle it
                            data, original_sr = sf.read(reference_audio)
                            waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data)
                    except Exception as sf_error:
                        logger.error(f"All audio loading methods failed. TorchCodec: {e}, Librosa: {librosa_error}, Soundfile: {sf_error}")
                        raise ImportError(
                            "Unable to load audio. Please install torchcodec or ensure librosa/soundfile is available. "
                            f"Original error: {e}, Librosa error: {librosa_error}, Soundfile error: {sf_error}"
                        )
            else:
                raise

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=sr
            )
            waveform = resampler(waveform)

        audio = waveform.squeeze().numpy()
        return audio
