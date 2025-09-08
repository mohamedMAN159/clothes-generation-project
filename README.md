# 🎨 AI Image Generation Project @ beyond apps group

A comprehensive image generation toolkit featuring Stable Diffusion XL (SDXL), FLUX, and custom LoRA fine-tuning capabilities. Generate high-quality images using various models with an easy-to-use web interface.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guides](#usage-guides)
  - [SDXL Local Generation](#sdxl-local-generation)
  - [FLUX API Generation](#flux-api-generation)
  - [Web UI](#web-ui)
  - [API Backend](#api-backend)
- [Fine-tuning Guide](#fine-tuning-guide)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration) 

## ✨ Features

- **Multiple Model Support**: SDXL, FLUX.1-schnell, and custom LoRA models
- **Web Interface**: Beautiful Streamlit UI for easy image generation
- **API Backend**: FastAPI server with ngrok tunneling
- **Local Generation**: Run models completely offline
- **Fine-tuning**: Train custom LoRA adapters on your data
- **Batch Processing**: Generate multiple images efficiently
- **Product Photography**: Specialized prompts for clothing/product images

## 🔧 Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (recommended: 8GB+ VRAM)
- **20GB+ free disk space** for models
- **Hugging Face account** (for FLUX API access)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1060 6GB | RTX 4060+ |
| RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ SSD |
| VRAM | 6GB | 12GB+ |

## 📦 Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd image-generation-project

# Create virtual environment
python -m venv ai_image_env
source ai_image_env/bin/activate  # Linux/Mac
# or
ai_image_env\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate peft
pip install fastapi uvicorn streamlit requests pillow
pip install datasets matplotlib tqdm numpy
pip install pyngrok  # Optional: for public API access
```

### 2. Download Models

#### Option A: Download SDXL Locally (Recommended)
```bash
# Create models directory
mkdir -p models
cd models

# Download SDXL base model
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
```

#### Option B: Use Hugging Face Auto-download
The scripts will automatically download models if not found locally.

### 3. Setup Hugging Face Token (for FLUX)

Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
# Set environment variable
export HF_TOKEN="your_token_here"

# Or create .env file
echo "HF_TOKEN=your_token_here" > .env
```

## 🚀 Quick Start

### Generate Your First Image

1. **Test SDXL locally**:
```bash
python test.py
```

2. **Test FLUX via API**:
```bash
python test_flux.py
```

3. **Start the web interface**:
```bash
# Terminal 1: Start API server
python main.py

# Terminal 2: Start web UI
streamlit run ui_sdxl.py
```

Visit `http://localhost:8501` in your browser!

## 📖 Usage Guides

### 🎨 SDXL Local Generation

Generate high-quality images using Stable Diffusion XL locally.

#### Basic Usage

```bash
python test.py
```

#### Custom Prompts

Edit the `prompt_configs` list in `test.py`:

```python
prompt_configs = [
    {
        "prompt": "your custom prompt here, highly detailed, 8k",
        "negative": "blurry, low quality, distorted"
    },
    # Add more prompts...
]
```

#### Configuration Options

In `test.py`, modify these parameters:
```python
num_inference_steps = 30    # Quality vs speed (10-100)
guidance_scale = 7.5        # Prompt adherence (1-20)
height = 1024              # Image height
width = 1024               # Image width
```

### 🌊 FLUX API Generation

Use FLUX.1-schnell for ultra-fast generation via Hugging Face API.

#### Setup

1. Get HF token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Set token in environment or edit `test_flux.py`

#### Custom Prompts

Edit the `prompt_configs` in `test_flux.py`:

```python
prompt_configs = [
    {
        "prompt": "your amazing prompt, professional photography",
        "negative": "amateur, blurry, low quality"
    }
]
```

#### Run Generation

```bash
python test_flux.py
```

### 🖥️ Web UI

Beautiful Streamlit interface for interactive image generation.

#### Start the UI

```bash
# Start API backend first
python main.py

# In another terminal, start UI
streamlit run ui_sdxl.py
```

#### Features

- Real-time parameter adjustment
- Image gallery with download
- Generation history tracking
- Preset prompt library
- Batch generation support

### 🔧 API Backend

FastAPI server with automatic ngrok tunneling for remote access.

#### Start Server

```bash
python main.py
```

#### Configuration

Edit the `Config` class in `main.py`:

```python
class Config:
    MODEL_PATH = "path/to/your/sdxl/model"
    LORA_PATH = "outputs/lora-cloth-v3/final_lora"
    HOST = "0.0.0.0"
    PORT = 8000
    DEVICE = "cuda"  # or "cpu"
```

#### API Endpoints

- `GET /`: Server status and info
- `GET /health`: Health check
- `POST /generate`: Generate images

#### Example API Request

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "beautiful landscape, highly detailed",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "lora_scale": 0.8
})

result = response.json()
```

## 🎓 Fine-tuning Guide

### Prepare Your Dataset

1. **Organize Images**:
```
dataset/
├── image_001.jpg
├── image_002.jpg
└── ...
```

2. **Create Captions**:
Create a `metadata.json` file:
```json
[
    {"file_name": "image_001.jpg", "text": "description of image 1"},
    {"file_name": "image_002.jpg", "text": "description of image 2"}
]
```

3. **Process Dataset**:
```python
from datasets import Dataset
import os

# Load your images and captions
data = []
for item in metadata:
    data.append({
        "image_path": os.path.join("dataset", item["file_name"]),
        "caption": item["text"]
    })

# Save processed dataset
dataset = Dataset.from_list(data)
dataset.save_to_disk("cloth_processed")
```

### Training Pipeline

#### Stage 1: Initial Fine-tuning

```bash
# Train base model (modify train_final.py for initial training)
python train_final.py
```

#### Stage 2: LoRA Fine-tuning

1. **Update paths** in `train_final.py`:
```python
previous_model_path = "path/to/your/trained/model"
dataset_dir = "cloth_processed"
```

2. **Run training**:
```bash
python train_final.py
```

#### Training Configuration

Key parameters in `train_final.py`:

```python
output_dir = "outputs/lora-cloth-v3"
num_epochs = 3
learning_rate = 1e-4
batch_size = 1
gradient_accumulation_steps = 4

# LoRA configuration
lora_config = LoraConfig(
    r=8,                    # Rank (complexity)
    lora_alpha=16,          # Scaling factor
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
)
```

### Using Your Trained Model

After training, use your model:

```python
from diffusers import StableDiffusionPipeline
import torch

# Load your trained model
pipe = StableDiffusionPipeline.from_pretrained(
    "outputs/lora-cloth-v3/final_model",
    torch_dtype=torch.float16,
    safety_checker=None
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# Generate images
image = pipe(
    "your custom prompt",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("custom_generated.png")
```

## 📁 Project Structure

```
image-generation-project/
├── main.py                 # FastAPI backend server
├── test.py                 # SDXL local testing script
├── test_flux.py           # FLUX API testing script
├── train_final.py         # LoRA fine-tuning script
├── ui_sdxl.py            # Streamlit web interface
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── models/               # Downloaded model files
│   └── stable-diffusion-xl-base-1.0/
├── outputs/              # Training outputs
│   └── lora-cloth-v3/
├── cloth_processed/      # Processed training dataset
└── generated_images/     # Output images
```

## 🔄 Workflow Examples

### 1. Product Photography Workflow

```bash
# 1. Generate product images with SDXL
python test.py

# 2. Or use FLUX for faster generation
python test_flux.py

# 3. Use web UI for interactive generation
streamlit run ui_sdxl.py
```

### 2. Custom Model Training Workflow

```bash
# 1. Prepare your dataset in cloth_processed/
# 2. Train LoRA adapter
python train_final.py

# 3. Update model path in main.py
# 4. Start API with your trained model
python main.py

# 5. Use web UI with your custom model
streamlit run ui_sdxl.py
```

## 🛠️ Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size in train_final.py
batch_size = 1

# Enable memory optimizations in test.py
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
```

#### Model Not Found
```bash
# Update model paths in configuration
MODEL_PATH = "D:/conda_FT/models/stable-diffusion-xl-base-1.0"
```

#### API Connection Issues
```bash
# Check if server is running
curl http://localhost:8000/health

# Or check in browser
http://localhost:8000
```

#### Package Installation Issues
```bash
# Install with conda instead
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Or use CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Performance Optimization

#### For RTX 4060 (8GB VRAM)
```python
# Enable memory efficient attention
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
```

#### For Lower VRAM GPUs
```python
# Use sequential CPU offloading
pipe.enable_sequential_cpu_offload()

# Reduce resolution
width, height = 512, 512  # Instead of 1024x1024
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:
```bash
HF_TOKEN=your_huggingface_token
MODEL_PATH=/path/to/your/models
CUDA_VISIBLE_DEVICES=0
```

### Model Paths

Update paths in each script:

**main.py**:
```python
class Config:
    MODEL_PATH = "D:/conda_FT/models/stable-diffusion-xl-base-1.0"
    LORA_PATH = "outputs/lora-cloth-v3/final_lora"
```

**test.py**:
```python
BASE_MODEL = r"D:\conda_FT\models\stable-diffusion-xl-base-1.0"
```

**train_final.py**:
```python
previous_model_path = r"D:\conda_FT\outputs\lora-cloth-incremental\final_lora"
```

## 🎯 Example Prompts

### Product Photography
```
"Clean white cotton t-shirt laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows"
```

### Artistic Generation
```
"Majestic dragon soaring through storm clouds, fantasy art, highly detailed, epic composition, dramatic lighting, 8k resolution"
```

### Portrait Photography
```
"Professional headshot of a business person, studio lighting, clean background, sharp focus, commercial photography"
```

## 📝 License

This project is for educational and research purposes. Please respect the licenses of the underlying models:
- Stable Diffusion XL: CreativeML Open RAIL++-M License
- FLUX.1: Apache 2.0 License

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Verify your CUDA installation: `nvidia-smi`
3. Check GPU memory: `torch.cuda.get_device_properties(0)`
4. Review error logs in the console output

## 🔗 Useful Links

- [Stable Diffusion Documentation](https://huggingface.co/docs/diffusers)
- [FLUX Model Page](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [LoRA Fine-tuning Guide](https://huggingface.co/docs/peft)
- [Hugging Face Tokens](https://huggingface.co/settings/tokens)

---


**Happy generating! 🎨**

