#!/usr/bin/env python3
"""
Test Script for Local SDXL Model with Negative Prompts
✅ Loads local SDXL model | ✅ Generates images with positive and negative prompts
"""

import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def test_sdxl_model():
    # Update this path to your SDXL model location
    BASE_MODEL = r"D:\conda_FT\models\stable-diffusion-xl-base-1.0"
    
    # Alternative common paths you might have:
    # BASE_MODEL = r"D:\conda_FT\models\stabilityai--stable-diffusion-xl-base-1.0"
    # BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # For HuggingFace download

    if not os.path.exists(BASE_MODEL):
        print(f"⚠️ Local model not found at: {BASE_MODEL}")
        print("🔄 Attempting to download from HuggingFace...")
        BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    # Enhanced prompts with more detailed descriptions
    prompt_configs = [
    {
        "prompt": "Clean white cotton t-shirt laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, body, mannequin, model, torso, arms, hands, face, skin, wearing, on body"
    },
    {
        "prompt": "Vintage blue denim jacket laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, model, body parts, mannequin, torso, shoulders, arms, wearing, on person"
    },
    {
        "prompt": "Elegant black evening dress laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, model, body, mannequin, torso, legs, arms, wearing, on body, face"
    },
    {
        "prompt": "athletic sneakers laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "feet, person, human, legs, body parts, wearing, on foot, model, mannequin"
    },
    {
        "prompt": "beige sweater laid flat on white marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, body, torso, arms, wearing, model, mannequin, on body"
    },
    {
        "prompt": "Navy blue business suit jacket laid flat on white marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, model, body, torso, shoulders, arms, wearing, mannequin, on person"
    },
    {
        "prompt": "summer dress laid flat on white marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, model, body, torso, legs, wearing, mannequin, on body, face"
    },
    {
        "prompt": "Brown leather jacket laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "feet, legs, person, human, wearing, on foot, body parts, model, mannequin"
    },
    {
        "prompt": "Dark blue shirt laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, neck, body parts, wearing, on person, model, mannequin, face"
    },
    {
        "prompt": "Blue denim pair of jeans trousers laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, legs, body, torso, wearing, on body, model, mannequin"
    },
    {
        "prompt": "black T-shirt laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, model, body, torso, arms, wearing, mannequin, on person"
    },
    {
        "prompt": "blue hoodie laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, body, torso, legs, arms, wearing, model, mannequin, on body"
    },
    {
        "prompt": "black leather pair of shoes laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "feet, legs, person, human, wearing, on foot, body parts, model, mannequin, more than one pair shoes"
    },
    {
        "prompt": "gray hoodie laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, model, body, torso, arms, head, wearing, mannequin, on body"
    },
    {
        "prompt": "leather handbag laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, hands, arms, body parts, carrying, holding, wearing, model, mannequin"
    },
    {
        "prompt": "black shirt laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, body, torso, chest, wearing, model, mannequin, on body, skin"
    },
    {
        "prompt": "White polo T-shirt laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, chef, body, torso, head, wearing, model, mannequin, on person"
    },
    {
        "prompt": "Blue dress laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, model, body, torso, legs, wearing, mannequin, on body, face"
    },
    {
        "prompt": "green hoodie laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "feet, legs, person, human, wearing, on foot, body parts, worker, model, mannequin"
    },
    {
        "prompt": "Black denim pair of jeans trousers laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, model, body, torso, wearing, mannequin, on body, fan, concert-goer"
    }
]
    # SDXL specific parameters (higher resolution support)
    num_inference_steps = 30  # Slightly higher for SDXL
    guidance_scale = 7.5
    height = 1024  # SDXL native resolution
    width = 1024   # SDXL native resolution
    generator = torch.manual_seed(42)

    print(f"🔄 Loading SDXL model from: {BASE_MODEL}")
    
    try:
        # Load SDXL pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16" if BASE_MODEL.startswith("stabilityai") else None
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ XFormers memory efficient attention enabled")
            except:
                pass
        
        print(f"✅ SDXL model loaded successfully on {device.upper()}")
        print(f"📏 Generation resolution: {width}x{height}")

    except Exception as e:
        print(f"❌ Error loading SDXL model: {str(e)}")
        return

    # Create output directory
    output_dir = "sdxl_test_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🎨 Generating {len(prompt_configs)} images using SDXL with negative prompts...")

    generated_images = []
    for i, config in enumerate(prompt_configs, 1):
        prompt = config["prompt"]
        negative_prompt = config["negative"]
        
        print(f"\n[{i}/{len(prompt_configs)}] Prompt: {prompt[:60]}...")
        print(f"    Negative: {negative_prompt[:60]}...")
        
        try:
            # Generate image with positive and negative prompts
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images

            for j, img in enumerate(images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sdxl_result_{i:02d}_{j+1}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                print(f"💾 Saved: {filepath}")
                generated_images.append((prompt, negative_prompt, img))

        except Exception as e:
            print(f"❌ Generation failed for prompt {i}: {e}")
            print(f"   Details: {type(e).__name__} - {e}")
            continue

    # Display results
    if generated_images:
        print(f"\n📊 Displaying {len(generated_images)} generated images...")

        cols = 2
        rows = (len(generated_images) + 1) // 2
        plt.figure(figsize=(16, 8 * rows))

        for idx, (prompt, neg_prompt, img) in enumerate(generated_images):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(img)
            short_prompt = prompt[:40] + "..." if len(prompt) > 40 else prompt
            short_neg = neg_prompt[:30] + "..." if len(neg_prompt) > 30 else neg_prompt
            plt.title(f"[{idx+1}] {short_prompt}\nNeg: {short_neg}", fontsize=8)
            plt.axis("off")

        summary_path = os.path.join(output_dir, f"sdxl_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.tight_layout()
        plt.savefig(summary_path, dpi=200, bbox_inches='tight')
        print(f"📈 Summary saved: {summary_path}")

        try:
            plt.show()
        except:
            print("⚠️ Cannot display images directly. Check 'sdxl_test_results' folder.")

    # Generate comparison: with vs without negative prompts
    print(f"\n🔄 Generating comparison images (with/without negative prompts)...")
    comparison_prompt = "Beautiful elegant dress, high fashion photography, professional lighting"
    comparison_negative = "blurry, low quality, deformed, ugly, poorly drawn, amateur"
    
    try:
        # Without negative prompt
        img_without_neg = pipe(
            prompt=comparison_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(123),
        ).images[0]
        
        # With negative prompt
        img_with_neg = pipe(
            prompt=comparison_prompt,
            negative_prompt=comparison_negative,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(123),
        ).images[0]
        
        # Save comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        axes[0].imshow(img_without_neg)
        axes[0].set_title("Without Negative Prompt", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(img_with_neg)
        axes[1].set_title("With Negative Prompt", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle(f"Comparison: {comparison_prompt}", fontsize=14, fontweight='bold')
        comparison_path = os.path.join(output_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
        print(f"🔍 Comparison saved: {comparison_path}")
        
        # Save individual comparison images
        img_without_neg.save(os.path.join(output_dir, "comparison_without_negative.png"))
        img_with_neg.save(os.path.join(output_dir, "comparison_with_negative.png"))
        
    except Exception as e:
        print(f"❌ Comparison generation failed: {e}")

    print("\n🎉 SDXL testing completed successfully!")
    print(f"📁 All results saved in: {output_dir}")

def print_system_info():
    """Print system and GPU information"""
    print("🖥️  System Information")
    print(f"- GPU Support: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"  - CUDA Version: {torch.version.cuda}")
    print(f"- PyTorch Version: {torch.__version__}")
    
    try:
        import diffusers
        print(f"- Diffusers Version: {diffusers.__version__}")
    except ImportError:
        print("- Diffusers: Not installed")

if __name__ == "__main__":
    print_system_info()
    print("\n" + "="*60)
    test_sdxl_model()