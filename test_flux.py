import requests
import os
from io import BytesIO
from PIL import Image
import json
import time
from pathlib import Path
from datetime import datetime

# Your prompt configurations
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
        "prompt": "Blue denim jeans laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
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
        "prompt": "black leather shoes laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "feet, legs, person, human, wearing, on foot, body parts, model, mannequin"
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
        "prompt": "Black denim jeans laid flat on marble surface, studio lighting, minimalist composition, high quality product photography, crisp details, soft shadows",
        "negative": "person, human, model, body, torso, wearing, mannequin, on body, fan, concert-goer"
    }
]

class HuggingFaceFluxGenerator:
    """Simple FLUX image generator using only Hugging Face API"""
    
    def __init__(self, api_key=None, output_dir="./flux_generated_images"):
        # Get API key from environment or parameter
        # Try multiple environment variable names and the hardcoded token as fallback
        self.api_key = (api_key or 
                       os.getenv("HF_TOKEN") or 
                       os.getenv("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") or
                       "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")
        
        if not self.api_key:
            raise ValueError("❌ API key not found! Set HF_TOKEN environment variable or pass api_key parameter")
        
        self.api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats tracking
        self.successful_generations = 0
        self.failed_generations = 0
        self.start_time = None

    def generate_image(self, prompt, width=1024, height=1024, num_inference_steps=4, guidance_scale=0.0):
        """Generate a single image using Hugging Face API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                return image
            elif response.status_code == 503:
                # Model is loading
                try:
                    result = response.json()
                    wait_time = result.get("estimated_time", 60)
                    print(f"⏳ Model loading, waiting {wait_time}s...")
                    time.sleep(min(wait_time, 120))  # Cap wait time at 2 minutes
                    # Retry once
                    return self.generate_image(prompt, width, height, num_inference_steps, guidance_scale)
                except:
                    print(f"⏳ Model loading, waiting 60s...")
                    time.sleep(60)
                    return self.generate_image(prompt, width, height, num_inference_steps, guidance_scale)
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ Request timed out after 120 seconds")
            return None
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

    def save_image(self, image, prompt, index):
        """Save image with descriptive filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract item name from prompt for filename
        words = prompt.lower().split()
        item_words = []
        for word in words:
            if word not in ['laid', 'flat', 'on', 'marble', 'surface', 'studio', 'lighting']:
                item_words.append(word)
                if len(item_words) >= 3:
                    break
        
        item_name = "_".join(item_words).replace(",", "")
        filename = f"{index+1:02d}_{item_name}_{timestamp}.png"
        filepath = self.output_dir / filename
        
        # Save with high quality
        image.save(filepath, "PNG", quality=95, optimize=True)
        return filepath

    def generate_batch(self, max_images=20, delay_between=3):
        """Generate batch of images from prompt configs"""
        
        print("🚀 Starting FLUX Batch Generation (Hugging Face Only)")
        print("=" * 60)
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"🎯 Target: {min(max_images, len(prompt_configs))} images")
        print(f"⏱️  Delay between requests: {delay_between}s")
        
        self.start_time = time.time()
        
        # Generate images
        total_to_generate = min(max_images, len(prompt_configs))
        
        for i in range(total_to_generate):
            config = prompt_configs[i]
            prompt = config["prompt"]
            
            print(f"\n🎨 Generating image {i + 1}/{total_to_generate}")
            print(f"📝 {prompt[:80]}...")
            
            try:
                # Generate image
                image = self.generate_image(prompt)
                
                if image:
                    # Save image
                    filepath = self.save_image(image, prompt, i)
                    print(f"✅ Saved: {filepath.name}")
                    self.successful_generations += 1
                    
                    # Show progress
                    print(f"📊 Progress: {self.successful_generations}/{total_to_generate} successful")
                else:
                    print(f"❌ Failed to generate image {i + 1}")
                    self.failed_generations += 1
                
                # Delay before next request (except for last image)
                if i < total_to_generate - 1:
                    print(f"⏸️  Waiting {delay_between}s before next generation...")
                    time.sleep(delay_between)
                    
            except KeyboardInterrupt:
                print("\n⏹️  Generation stopped by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error for image {i+1}: {e}")
                self.failed_generations += 1
                continue
        
        self.print_summary()

    def print_summary(self):
        """Print generation summary"""
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 60)
        print("📊 GENERATION COMPLETE!")
        print("=" * 60)
        print(f"✅ Successful generations: {self.successful_generations}")
        print(f"❌ Failed generations: {self.failed_generations}")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print(f"📁 Images saved to: {self.output_dir.absolute()}")
        
        if self.successful_generations > 0:
            avg_time = total_time / self.successful_generations
            print(f"📈 Average time per image: {avg_time:.1f} seconds")
            
            # Show total disk space used
            total_size = sum(f.stat().st_size for f in self.output_dir.glob("*.png"))
            total_size_mb = total_size / (1024 * 1024)
            print(f"💾 Total disk space used: {total_size_mb:.1f} MB")
            
            print(f"\n📂 Generated files:")
            for i, file in enumerate(sorted(self.output_dir.glob("*.png")), 1):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   {i:2d}. {file.name} ({size_mb:.1f} MB)")

def main():
    """Main function"""
    print("🏪 HuggingFace FLUX Product Image Generator")
    print("=" * 60)
    
    # Check for API key
    api_key = (os.getenv("HF_TOKEN") or 
               os.getenv("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") or
               "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")
    
    if not api_key or api_key == "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":
        print("✅ Using hardcoded API key")
    else:
        print(f"✅ API key found: {api_key[:8]}..." + "*" * max(0, len(api_key) - 8))
    
    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"flux_product_images_{timestamp}"
    
    try:
        # Initialize generator
        generator = HuggingFaceFluxGenerator(output_dir=output_folder)
        
        # Show configuration
        print(f"\n⚙️  Configuration:")
        print(f"   📝 Total prompts available: {len(prompt_configs)}")
        print(f"   🎯 Images to generate: 20")
        print(f"   📁 Output folder: {generator.output_dir}")
        print(f"   🔧 Model: FLUX.1-schnell (fast, 4 steps)")
        
        # Get user preferences
        num_images = input(f"\nHow many images to generate? (1-{len(prompt_configs)}, default: 20): ").strip()
        try:
            num_images = int(num_images) if num_images else 20
            num_images = max(1, min(num_images, len(prompt_configs)))
        except ValueError:
            num_images = 20
        
        delay = input("Delay between generations in seconds (default: 3): ").strip()
        try:
            delay = int(delay) if delay else 3
            delay = max(1, delay)  # Minimum 1 second
        except ValueError:
            delay = 3
        
        # Final confirmation
        proceed = input(f"\n🚀 Generate {num_images} images with {delay}s delay? (y/n): ").strip().lower()
        if proceed != 'y':
            print("❌ Generation cancelled")
            return
        
        # Start generation
        print(f"\n🎨 Starting generation of {num_images} images...")
        generator.generate_batch(max_images=num_images, delay_between=delay)
        
        # Save generation log
        log_file = generator.output_dir / "generation_log.json"
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "successful_generations": generator.successful_generations,
            "failed_generations": generator.failed_generations,
            "output_directory": str(generator.output_dir.absolute()),
            "model_used": "FLUX.1-schnell",
            "api_used": "Hugging Face Inference API",
            "total_prompts_available": len(prompt_configs),
            "prompts_used": prompt_configs[:num_images]
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Generation log saved to: {log_file}")
        
    except ValueError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()