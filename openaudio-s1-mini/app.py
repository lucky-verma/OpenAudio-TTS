import os
import queue
from huggingface_hub import snapshot_download
import numpy as np
import wave
import io
import gc
from typing import Callable

# Download if not exists
# Handle permission issues on Windows mounts by trying multiple locations
checkpoints_base = "checkpoints"
checkpoints_path = f"{checkpoints_base}/openaudio-s1-mini"

# Try to create checkpoints directory, fallback to home directory if permission denied
try:
    os.makedirs(checkpoints_base, exist_ok=True)
except PermissionError:
    # If we can't create in current directory (Windows mount), use home directory
    import pathlib
    home_checkpoints = pathlib.Path.home() / "openaudio_checkpoints"
    checkpoints_base = str(home_checkpoints)
    checkpoints_path = str(home_checkpoints / "openaudio-s1-mini")
    os.makedirs(checkpoints_base, exist_ok=True)
    print(f"⚠ Permission denied in current directory. Using: {checkpoints_base}")

# Check if checkpoints already exist and are complete
checkpoints_exist = os.path.exists(checkpoints_path) and os.listdir(checkpoints_path)
checkpoints_complete = checkpoints_exist and os.path.exists(os.path.join(checkpoints_path, "config.json"))

if not checkpoints_complete:
    if checkpoints_exist:
        print(f"⚠ Checkpoints directory exists but appears incomplete at: {checkpoints_path}")
        print("  Re-downloading to ensure all files are present...")
    else:
        print("Downloading model checkpoints...")
    
    print("⚠ NOTE: This model requires Hugging Face authentication.")
    print("   Make sure you have:")
    print("   1. Accepted the model terms at: https://huggingface.co/fishaudio/openaudio-s1-mini")
    print("   2. Logged in with: huggingface-cli login")
    print()
    try:
        snapshot_download(repo_id="fishaudio/openaudio-s1-mini", local_dir=checkpoints_path)
        print("✓ All checkpoints downloaded")
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e) or "GatedRepoError" in str(e):
            print("\n❌ Authentication Error!")
            print("Please authenticate with Hugging Face:")
            print("  1. Visit: https://huggingface.co/fishaudio/openaudio-s1-mini")
            print("  2. Accept the model terms and conditions")
            print("  3. Run: huggingface-cli login")
            print("  4. Then run this script again")
            raise
        else:
            raise
else:
    print(f"✓ Checkpoints already exist and appear complete at: {checkpoints_path}")

# Store the checkpoint path for use in parse_args
CHECKPOINT_PATH = checkpoints_path

import html
import os
from argparse import ArgumentParser
from pathlib import Path

import gradio as gr
import torch
import torchaudio

# Set audio backend if the method exists (older versions may not have it)
try:
    torchaudio.set_audio_backend("soundfile")
except AttributeError:
    # torchaudio.set_audio_backend doesn't exist in this version
    # soundfile backend should be used by default
    pass

from loguru import logger
from fish_speech.i18n import i18n
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from tools.webui.inference import get_inference_wrapper
from fish_speech.utils.schema import ServeTTSRequest

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


HEADER_MD = """# OpenAudio S1

## The demo in this space is OpenAudio S1, Please check [Fish Audio](https://fish.audio) for the best model.
## 该 Demo 为 OpenAudio S1 版本, 请在 [Fish Audio](https://fish.audio) 体验最新 DEMO.

A text-to-speech model based on DAC & Qwen3 developed by [Fish Audio](https://fish.audio).  
由 [Fish Audio](https://fish.audio) 研发的 DAC & Qwen3 多语种语音合成. 

You can find the source code [here](https://github.com/fishaudio/fish-speech) and models [here](https://huggingface.co/fishaudio/openaudio-s1-mini).  
你可以在 [这里](https://github.com/fishaudio/fish-speech) 找到源代码和 [这里](https://huggingface.co/fishaudio/openaudio-s1-mini) 找到模型.  

Related code and weights are released under CC BY-NC-SA 4.0 License.  
相关代码，权重使用 CC BY-NC-SA 4.0 许可证发布.

We are not responsible for any misuse of the model, please consider your local laws and regulations before using it.  
我们不对模型的任何滥用负责，请在使用之前考虑您当地的法律法规.

The model running in this WebUI is OpenAudio S1 Mini.
在此 WebUI 中运行的模型是 OpenAudio S1 Mini.
"""

TEXTBOX_PLACEHOLDER = """Put your text here. 在此处输入文本."""

try:
    import spaces

    GPU_DECORATOR = spaces.GPU
except ImportError:

    def GPU_DECORATOR(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

def build_html_error_message(error):
    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """

def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


def build_app(inference_fct: Callable, theme: str = "light") -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=10
                )

                with gr.Row():
                    with gr.Column():
                        with gr.Tab(label=i18n("Advanced Config")):
                            with gr.Row():
                                chunk_length = gr.Slider(
                                    label=i18n("Iterative Prompt Length, 0 means off"),
                                    minimum=0,
                                    maximum=500,
                                    value=0,
                                    step=8,
                                )

                                max_new_tokens = gr.Slider(
                                    label=i18n(
                                        "Maximum tokens per batch, 0 means no limit"
                                    ),
                                    minimum=0,
                                    maximum=2048,
                                    value=0,
                                    step=8,
                                )

                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-P",
                                    minimum=0.7,
                                    maximum=0.95,
                                    value=0.9,
                                    step=0.01,
                                )

                                repetition_penalty = gr.Slider(
                                    label=i18n("Repetition Penalty"),
                                    minimum=1,
                                    maximum=1.2,
                                    value=1.1,
                                    step=0.01,
                                )

                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.7,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.01,
                                )
                                seed = gr.Number(
                                    label="Seed",
                                    info="0 means randomized inference, otherwise deterministic",
                                    value=0,
                                )
                            
                            with gr.Row():
                                normalize = gr.Checkbox(
                                    label=i18n("Text Normalization"),
                                    info=i18n("Normalize numbers and text (converts numbers to Chinese numerals). Disable for English text with numbers."),
                                    value=False,
                                )

                        with gr.Tab(label=i18n("Reference Audio")):
                            with gr.Row():
                                gr.Markdown(
                                    i18n(
                                        "5 to 10 seconds of reference audio, useful for specifying speaker."
                                    )
                                )
                            with gr.Row():
                                reference_id = gr.Textbox(
                                    label=i18n("Reference ID"),
                                    placeholder="Leave empty to use uploaded references",
                                )

                            with gr.Row():
                                use_memory_cache = gr.Radio(
                                    label=i18n("Use Memory Cache"),
                                    choices=["on", "off"],
                                    value="on",
                                )

                            with gr.Row():
                                reference_audio = gr.Audio(
                                    label=i18n("Reference Audio"),
                                    type="filepath",
                                )
                            with gr.Row():
                                reference_text = gr.Textbox(
                                    label=i18n("Reference Text"),
                                    lines=1,
                                    placeholder="在一无所知中，梦里的一天结束了，一个新的「轮回」便会开始。",
                                    value="",
                                )

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001f3a7 " + i18n("Generate"),
                            variant="primary",
                        )

        # Submit
        generate.click(
            inference_fct,
            [
                text,
                reference_id,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache,
                normalize,
            ],
            [audio, error],
            concurrency_limit=1,
        )

    return app

def parse_args():
    parser = ArgumentParser()
    # Use the checkpoint path determined during download
    # This ensures consistency between download location and model loading
    # First check if CHECKPOINT_PATH was set (from the download section)
    if 'CHECKPOINT_PATH' in globals():
        default_checkpoint = CHECKPOINT_PATH
    else:
        # Fallback: check if default exists, otherwise try home directory
        default_checkpoint = "checkpoints/openaudio-s1-mini"
        if not os.path.exists(default_checkpoint):
            home_checkpoint = os.path.expanduser("~/openaudio_checkpoints/openaudio-s1-mini")
            if os.path.exists(home_checkpoint):
                default_checkpoint = home_checkpoint
    
    # Verify the checkpoint path exists and has required files
    if not os.path.exists(default_checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {default_checkpoint}")
    
    # Check for config.json to verify it's a valid checkpoint directory
    config_path = os.path.join(default_checkpoint, "config.json")
    if not os.path.exists(config_path):
        # Maybe it's a different structure, check for any .json files
        import glob
        json_files = glob.glob(os.path.join(default_checkpoint, "*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"Invalid checkpoint directory: {default_checkpoint}\n"
                f"Expected config.json not found. Found files: {os.listdir(default_checkpoint) if os.path.exists(default_checkpoint) else 'directory does not exist'}"
            )
    
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default=default_checkpoint,
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default=f"{default_checkpoint}/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true",default=True)
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--theme", type=str, default="dark")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine precision: check if GPU supports bfloat16 natively
    if args.half:
        args.precision = torch.half
    elif args.device == "cuda" and torch.cuda.is_available():
        # Check if GPU supports bfloat16 natively (Ampere+ architecture)
        # RTX 2070 SUPER (Turing) doesn't support bfloat16 natively
        # RTX 30xx/40xx (Ampere/Ada) do support it
        try:
            # Try to get compute capability
            compute_capability = torch.cuda.get_device_capability()
            # Ampere (8.0+) supports bfloat16 natively
            if compute_capability[0] >= 8:
                args.precision = torch.bfloat16
            else:
                # Older GPUs (Turing, Pascal, etc.) - use float16
                args.precision = torch.half
                logger.info(f"GPU compute capability {compute_capability[0]}.{compute_capability[1]} doesn't support bfloat16 natively, using float16")
        except Exception:
            # Fallback to float16 if we can't determine capability
            args.precision = torch.half
            logger.warning("Could not determine GPU capability, defaulting to float16")
    else:
        # CPU or no CUDA - use bfloat16 (CPU supports it)
        args.precision = torch.bfloat16

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )
    logger.info("Llama model loaded, loading VQ-GAN model...")

    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Decoder model loaded, warming up...")

    # Create the inference engine
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
        inference_engine.inference(
            ServeTTSRequest(
                text="Hello world.",
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
            )
        )
    )

    logger.info("Warming up done, launching the web UI...")

    inference_fct = get_inference_wrapper(inference_engine)

    app = build_app(inference_fct, args.theme)
    app.queue(api_open=True).launch(show_error=True, show_api=True)
