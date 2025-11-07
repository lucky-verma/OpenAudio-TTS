# Sample Generation Scripts

This directory contains utility scripts for generating TTS samples.

## generate_samples.py

Generates a variety of TTS samples with different emotions and configurations.

### Usage

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Run the script
python scripts/generate_samples.py
```

### Options

- `--device`: Device to use (`cuda`, `cpu`, or `auto` - default: `auto`)
- `--checkpoint-path`: Path to model checkpoints (default: `checkpoints/openaudio-s1-mini`)
- `--output-dir`: Directory to save samples (default: `samples/outputs`)

### Example

```bash
python scripts/generate_samples.py --device cpu --output-dir my_samples
```

The script will generate:
- Basic TTS sample
- Samples with different emotions (excited, sad, angry)
- Samples with different tones (whispering, shouting)
- Multilingual sample

All samples are saved in a timestamped directory under the output directory.

