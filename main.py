# main.py
import os
import io
import base64
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from diffusers import StableDiffusionPipeline
from typing import List, Optional
import logging
import threading
from contextlib import asynccontextmanager
import gc

# Pyngrok for tunneling
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("⚠️  pyngrok not installed. Install with: pip install pyngrok")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Configuration
# -------------------------
class Config:
    MODEL_PATH = "D:\conda_FT\models\stable-diffusion-xl-base-1.0"
    LORA_PATH = "outputs/lora-cloth-v3/final_lora"
    HOST = "0.0.0.0"
    PORT = 8000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 7.5


# -------------------------
# Request/Response Models
# -------------------------
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    num_inference_steps: int = Field(Config.DEFAULT_STEPS, ge=1, le=100)
    guidance_scale: float = Field(Config.DEFAULT_GUIDANCE, ge=0.1, le=20.0)
    seed: Optional[int] = Field(None, description="Random seed")
    lora_scale: float = Field(0.8, ge=0.0, le=3.0, description="LoRA strength")


class GenerationResponse(BaseModel):
    success: bool
    images: List[str]  # base64 encoded PNGs
    metadata: dict
    processing_time: float
    message: Optional[str] = None


# -------------------------
# Model Manager
# -------------------------
class ModelManager:
    def __init__(self):
        self.pipeline = None
        self.model_loaded = False

    def load_model(self):
        try:
            logger.info(f"🚀 Loading base model: {Config.MODEL_PATH}")
            logger.info(f"🔧 Applying LoRA: {Config.LORA_PATH}")
            logger.info(f"💻 Device: {Config.DEVICE}")

            # Load base model
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                Config.MODEL_PATH,
                torch_dtype=Config.TORCH_DTYPE,
                safety_checker=None,
                requires_safety_checker=False
            )

            # Apply LoRA
            if os.path.exists(Config.LORA_PATH):
                self.pipeline.load_lora_weights(Config.LORA_PATH)
                logger.info("✅ LoRA weights loaded")
            else:
                logger.warning(f"⚠️ LoRA not found at {Config.LORA_PATH}")

            # Optimize for GPU
            if Config.DEVICE == "cuda":
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ XFormers enabled")
                except Exception as e:
                    logger.warning(f"⚠️ XFormers not available: {e}")

                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_attention_slicing(1)
            else:
                self.pipeline = self.pipeline.to(Config.DEVICE)

            self.model_loaded = True
            logger.info("🎉 Model loaded successfully!")

            # Test generation
            self._test()

        except Exception as e:
            logger.error(f"❌ Load failed: {e}")
            raise

    def _test(self):
        try:
            logger.info("🧪 Running test generation...")
            self.pipeline("test", num_inference_steps=2, width=128, height=128)
            logger.info("✅ Test passed")
        except Exception as e:
            logger.warning(f"⚠️ Test failed: {e}")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        import time
        start_time = time.time()
        try:
            if request.seed is not None:
                torch.manual_seed(request.seed)
                if Config.DEVICE == "cuda":
                    torch.cuda.manual_seed_all(request.seed)

            with torch.autocast(Config.DEVICE if Config.DEVICE != "cpu" else "cpu"):
                result = self.pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    cross_attention_kwargs={"scale": request.lora_scale},
                )

            images_b64 = []
            for img in result.images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                images_b64.append(img_b64)

            processing_time = time.time() - start_time
            return GenerationResponse(
                success=True,
                images=images_b64,
                metadata={
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "width": request.width,
                    "height": request.height,
                    "steps": request.num_inference_steps,
                    "guidance": request.guidance_scale,
                    "lora_scale": request.lora_scale,
                    "seed": request.seed
                },
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return GenerationResponse(
                success=False,
                images=[],
                metadata={},
                processing_time=time.time() - start_time,
                message=str(e)
            )
        finally:
            if Config.DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()


# -------------------------
# Global Instances
# -------------------------
model_manager = ModelManager()
ngrok_tunnel = None

def setup_ngrok():
    global ngrok_tunnel
    if not NGROK_AVAILABLE:
        return
    try:
        ngrok_tunnel = ngrok.connect(Config.PORT)
        logger.info(f"🌍 Public API: {ngrok_tunnel.public_url}")
        logger.info("💡 WAF is enforced via ngrok.yml — no further action needed")
    except Exception as e:
        logger.error(f"❌ ngrok failed: {e}")


# -------------------------
# FastAPI App
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up service...")
    model_manager.load_model()
    threading.Thread(target=setup_ngrok, daemon=True).start()
    yield
    logger.info("🛑 Shutting down...")
    if ngrok_tunnel:
        ngrok.disconnect(ngrok_tunnel.public_url)


app = FastAPI(
    title="🎨 DreamStudio API",
    description="LoRA-powered image generator with ngrok WAF protection",
    version="1.0.0",
    lifespan=lifespan
)

# ✅ Allow Streamlit from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# API Endpoints
# -------------------------
@app.get("/")
def root():
    return {
        "message": "DreamStudio - LoRA API",
        "status": "running",
        "model": "SD 1.5 + LoRA",
        "waf": "Enabled via ngrok OWASP-CRS",
        "endpoints": ["/generate", "/health"],
        "ngrok_url": ngrok_tunnel.public_url if ngrok_tunnel else None
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_manager.model_loaded,
        "device": Config.DEVICE
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    if not model_manager.model_loaded:
        raise HTTPException(503, "Model not loaded yet")
    return model_manager.generate(request)


# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    mod = os.path.splitext(os.path.basename(__file__))[0]
    uvicorn.run(f"{mod}:app", host=Config.HOST, port=Config.PORT, reload=False, log_level="info")