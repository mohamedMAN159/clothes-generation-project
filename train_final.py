import os
import torch
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image
import numpy as np
import json

# -------------------------
# Configuration
# -------------------------
output_dir = "outputs/lora-cloth-v3"
num_epochs = 3
learning_rate = 1e-4
batch_size = 1
gradient_accumulation_steps = 4
mixed_precision = "fp16"  # Use 'fp16' for RTX 4060

# IMPORTANT: Update this path to your previously trained model
# This should be the path to your fine-tuned SD1.5 model directory
previous_model_path = r"D:\conda_FT\outputs\lora-cloth-incremental\final_lora"

# Dataset directory
dataset_dir = "cloth_processed"

os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Custom Dataset
# -------------------------
class ClothDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return {
            "image_path": example["image_path"],
            "caption": example["caption"]
        }

# -------------------------
# Collate Function
# -------------------------
def collate_fn(examples):
    images = [Image.open(e["image_path"]).convert("RGB") for e in examples]
    captions = [e["caption"] for e in examples]

    # Resize
    images = [img.resize((512, 512), Image.LANCZOS) for img in images]

    # Convert to tensor and normalize to [-1, 1]
    pixel_values = [torch.tensor(np.array(img)).permute(2, 0, 1) for img in images]
    pixel_values = torch.stack(pixel_values).to(torch.float16) / 127.5 - 1.0

    return {
        "pixel_values": pixel_values,
        "captions": captions
    }

# -------------------------
# Model Loading Functions
# -------------------------
def load_previous_model(model_path):
    """Load your previously trained model"""
    print(f"[INFO] Loading previously trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    try:
        # Load the full pipeline first
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        print("[SUCCESS] Previously trained model loaded successfully!")
        print(f"[INFO] Model components loaded:")
        print(f"  - UNet: {type(pipeline.unet).__name__}")
        print(f"  - VAE: {type(pipeline.vae).__name__}")
        print(f"  - Text Encoder: {type(pipeline.text_encoder).__name__}")
        
        return pipeline
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[INFO] Falling back to base SD1.5...")
        
        # Fallback to base model if loading fails
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        print("[WARNING] Using base SD1.5 instead of your trained model!")
        return pipeline

# -------------------------
# Local Model Saving Functions
# -------------------------
def save_lora_model(unet, epoch, output_dir, base_model_info):
    """Save LoRA adapter weights only"""
    lora_dir = os.path.join(output_dir, f"lora_epoch_{epoch + 1}")
    os.makedirs(lora_dir, exist_ok=True)
    
    # Save LoRA weights
    unet.save_pretrained(lora_dir)
    
    # Save config for easy loading
    config = {
        "base_model": previous_model_path,
        "base_model_info": base_model_info,
        "lora_rank": 8,
        "lora_alpha": 16,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        "lora_dropout": 0.1,
        "epoch": epoch + 1,
        "learning_rate": learning_rate,
        "incremental_training": True,
        "training_note": "This LoRA was trained on top of a previously fine-tuned model"
    }
    
    with open(os.path.join(lora_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print(f"[INFO] LoRA model saved at: {lora_dir}")
    return lora_dir

def save_full_local_model(accelerator, unet, vae, text_encoder, tokenizer, scheduler, epoch, output_dir, base_model_info):
    """Save complete model for local use"""
    try:
        # Get unwrapped UNet
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # Create a copy of the base UNet from your previous model
        print("[INFO] Creating merged model from your previously trained base...")
        
        # Load your previously trained UNet as base
        previous_pipeline = load_previous_model(previous_model_path)
        base_unet = previous_pipeline.unet
        
        # Apply LoRA config to the base UNet
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
        )
        temp_unet = get_peft_model(base_unet, lora_config)
        
        # Load the trained LoRA weights into the temp model
        temp_unet.load_state_dict(unwrapped_unet.state_dict())
        
        # Merge LoRA into your previously trained model
        merged_unet = temp_unet.merge_and_unload()
        
        # Create full pipeline using your previous model's components + new UNet
        pipeline = StableDiffusionPipeline(
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=merged_unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        # Save complete model
        full_model_dir = os.path.join(output_dir, f"full_model_epoch_{epoch + 1}")
        pipeline.save_pretrained(full_model_dir)
        
        # Save usage instructions
        usage_code = f"""
# How to use this incrementally trained model locally:

from diffusers import StableDiffusionPipeline
import torch

# Load the incrementally trained model
pipe = StableDiffusionPipeline.from_pretrained(
    "{os.path.abspath(full_model_dir)}",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)

# Move to GPU if available
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# Generate images
prompt = "your prompt here"
image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

# Save the image
image.save("generated_image.png")

# Note: This model has been trained incrementally on top of:
# Base model: {previous_model_path}
# Additional training epochs: {epoch + 1}
"""
        
        with open(os.path.join(full_model_dir, "usage_example.py"), "w", encoding="utf-8") as f:
            f.write(usage_code)
        
        # Save model info
        model_info = {
            "model_type": "Incremental LoRA Fine-tuning on Previously Trained SD1.5",
            "original_base_model": "runwayml/stable-diffusion-v1-5",
            "previous_model": previous_model_path,
            "previous_model_info": base_model_info,
            "incremental_epochs": epoch + 1,
            "total_training_stages": "Base SD1.5 → Previous Fine-tune → LoRA Incremental",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "lora_rank": 8,
            "lora_alpha": 16,
            "recommended_settings": {
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "height": 512,
                "width": 512
            },
            "training_note": "This model builds upon previous fine-tuning with additional LoRA layers"
        }
        
        with open(os.path.join(full_model_dir, "model_info.json"), "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"[SUCCESS] Incrementally trained model saved at: {full_model_dir}")
        print(f"[INFO] Usage example saved at: {os.path.join(full_model_dir, 'usage_example.py')}")
        
        # Clean up temporary models
        del previous_pipeline, base_unet, temp_unet, merged_unet, pipeline
        torch.cuda.empty_cache()
        
        return full_model_dir
        
    except Exception as e:
        print(f"[WARNING] Error saving full model: {e}")
        return None

def create_inference_script(output_dir):
    """Create a standalone inference script"""
    script_content = f'''
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse

def load_model(model_path):
    """Load the incrementally trained model"""
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("[SUCCESS] Incrementally trained model loaded on GPU")
        else:
            pipe = pipe.to("cpu")
            print("[SUCCESS] Incrementally trained model loaded on CPU")
            
        print("[INFO] This model was trained incrementally on: {previous_model_path}")
        return pipe
    except Exception as e:
        print(f"[ERROR] Error loading model: {{e}}")
        return None

def generate_image(pipe, prompt, output_path="generated.png", **kwargs):
    """Generate an image from prompt"""
    default_settings = {{
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "num_images_per_prompt": 1
    }}
    
    # Update with user settings
    settings = {{**default_settings, **kwargs}}
    
    print(f"[INFO] Generating image with prompt: '{{prompt}}'")
    print(f"[INFO] Settings: {{settings}}")
    
    try:
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            result = pipe(prompt, **settings)
        
        image = result.images[0]
        image.save(output_path)
        print(f"[SUCCESS] Image saved to: {{output_path}}")
        return image
        
    except Exception as e:
        print(f"[ERROR] Error generating image: {{e}}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate images with your incrementally trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to incrementally trained model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="generated.png", help="Output image path")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    
    args = parser.parse_args()
    
    # Load model
    pipe = load_model(args.model)
    if pipe is None:
        return
    
    # Generate image
    generate_image(
        pipe, 
        args.prompt, 
        args.output,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height
    )

if __name__ == "__main__":
    main()

# Example usage:
# python inference.py --model "path/to/your/incremental/model" --prompt "a beautiful cloth design"
'''
    
    script_path = os.path.join(output_dir, "inference.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"[INFO] Inference script created at: {script_path}")
    return script_path

# -------------------------
# Training Function
# -------------------------
def main():
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        print(f"[INFO] Starting incremental LoRA training for {num_epochs} epochs...")
        print(f"[INFO] Base model: {previous_model_path}")
        print(f"[INFO] Output directory: {os.path.abspath(output_dir)}")

    # Load dataset
    ds = load_from_disk(dataset_dir)
    train_dataset = ClothDataset(ds)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load your previously trained model
    previous_pipeline = load_previous_model(previous_model_path)
    
    # Extract components from your trained model
    tokenizer = previous_pipeline.tokenizer
    text_encoder = previous_pipeline.text_encoder
    vae = previous_pipeline.vae
    unet = previous_pipeline.unet
    scheduler = previous_pipeline.scheduler
    
    # Store base model info for later reference
    base_model_info = {
        "model_path": previous_model_path,
        "loaded_successfully": True,
        "model_type": type(previous_pipeline).__name__
    }

    if accelerator.is_main_process:
        print("[SUCCESS] Using your previously trained model as base!")

    # Freeze components (keep your trained weights frozen, only train LoRA)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)  # This will be unfrozen for LoRA parameters only

    # Apply LoRA to UNet (this will add trainable parameters on top of your model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    
    if accelerator.is_main_process:
        print("[INFO] LoRA applied to your previously trained UNet:")
        unet.print_trainable_parameters()

    # Create optimizer (only for LoRA parameters)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # Prepare models with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    
    vae = accelerator.prepare(vae)
    text_encoder = accelerator.prepare(text_encoder)

    # Cast to appropriate dtypes
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    # Verify trainable parameters
    trainable_count = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    if trainable_count == 0:
        raise ValueError("No trainable parameters found!")
    
    if accelerator.is_main_process:
        print(f"[SUCCESS] Trainable LoRA parameters: {trainable_count:,}")
        print(f"[INFO] Your base model parameters remain frozen")

    # Create inference script
    if accelerator.is_main_process:
        create_inference_script(output_dir)

    # Training loop
    progress_bar = tqdm(
        total=num_epochs * len(train_dataloader),
        disable=not accelerator.is_main_process
    )

    for epoch in range(num_epochs):
        unet.train()
        epoch_losses = []
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Prepare inputs
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)

                # Encode to latent space using your trained VAE
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Encode text using your trained text encoder
                inputs = tokenizer(
                    batch["captions"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = inputs.input_ids.to(accelerator.device)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # Add noise
                noise = torch.randn_like(latents, dtype=latents.dtype, device=latents.device)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (bsz,), 
                    device=latents.device, dtype=torch.long
                )
                
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Predict noise using your model + LoRA
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                if accelerator.is_main_process:
                    epoch_losses.append(loss.item())
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}", 
                        "epoch": epoch + 1,
                        "step": step + 1
                    })

        # Save models at end of each epoch
        if accelerator.is_main_process:
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            print(f"\n[INFO] Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Save LoRA weights
            unwrapped_unet = accelerator.unwrap_model(unet)
            save_lora_model(unwrapped_unet, epoch, output_dir, base_model_info)
            
            # Save full model for local use
            save_full_local_model(
                accelerator, unet, vae, text_encoder, 
                tokenizer, scheduler, epoch, output_dir, base_model_info
            )

    # Final save
    if accelerator.is_main_process:
        print("\n[INFO] Saving final incremental models...")
        
        # Save final LoRA
        unwrapped_unet = accelerator.unwrap_model(unet)
        final_lora_dir = os.path.join(output_dir, "final_lora")
        os.makedirs(final_lora_dir, exist_ok=True)
        unwrapped_unet.save_pretrained(final_lora_dir)
        
        # Save config
        final_config = {
            "base_model": previous_model_path,
            "base_model_info": base_model_info,
            "lora_rank": 8,
            "lora_alpha": 16,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "lora_dropout": 0.1,
            "total_epochs": num_epochs,
            "learning_rate": learning_rate,
            "final_model": True,
            "training_type": "incremental_lora",
            "training_note": "Final LoRA weights for incremental training on previously fine-tuned model"
        }
        
        with open(os.path.join(final_lora_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(final_config, f, indent=2)
        
        # Save final full model
        final_full_dir = save_full_local_model(
            accelerator, unet, vae, text_encoder, 
            tokenizer, scheduler, num_epochs - 1, output_dir, base_model_info
        )
        
        if final_full_dir:
            # Rename to final
            final_dir = os.path.join(output_dir, "final_model")
            if os.path.exists(final_dir):
                import shutil
                shutil.rmtree(final_dir)
            os.rename(final_full_dir, final_dir)
        
        print(f"\n[SUCCESS] Incremental training completed!")
        print(f"[INFO] Your model progression: Base SD1.5 → {previous_model_path} → {os.path.abspath(output_dir)}")
        print(f"[INFO] Models saved in: {os.path.abspath(output_dir)}")
        print(f"[INFO] Use 'python {os.path.join(output_dir, 'inference.py')}' to generate images")
        print(f"[EXAMPLE] python {os.path.join(output_dir, 'inference.py')} --model {os.path.join(output_dir, 'final_model')} --prompt 'beautiful cloth design'")

    progress_bar.close()

if __name__ == "__main__":
    main()