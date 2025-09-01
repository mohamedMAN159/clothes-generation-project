import streamlit as st
import requests
import base64
import io
from PIL import Image
import json
import time

# Page configuration
st.set_page_config(
    page_title="🎨 SDXL Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .generation-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #ff4757;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #2ed573;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'prompt_input' not in st.session_state:
    st.session_state.prompt_input = "a beautiful landscape with mountains and a lake, highly detailed, 8k resolution"

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

def generate_image(prompt, negative_prompt, width, height, steps, guidance, seed=None):
    """Generate image using the SDXL API"""
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": seed
    }
    
    try:
        with st.spinner("🎨 Generating your masterpiece..."):
            response = requests.post(f"{API_BASE_URL}/generate", json=payload, timeout=300)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

# Main UI
st.markdown('<h1 class="main-header">🎨 SDXL Image Generator</h1>', unsafe_allow_html=True)

# Check API status
is_healthy, health_data = check_api_health()

if not is_healthy:
    st.markdown(
        '<div class="error-box">❌ API Server is not running! Please start the SDXL API server first.</div>',
        unsafe_allow_html=True
    )
    st.code("python main.py", language="bash")
    st.stop()
else:
    st.markdown(
        f'<div class="success-box">✅ API Server is running on {health_data.get("device", "unknown")} device</div>',
        unsafe_allow_html=True
    )

# Sidebar for parameters
st.sidebar.header("🎛️ Generation Parameters")

# Use session state for prompt
prompt = st.sidebar.text_area(
    "Prompt",
    value=st.session_state.get('prompt_input', "a beautiful landscape with mountains and a lake, highly detailed, 8k resolution"),
    height=100,
    help="Describe what you want to generate",
    key="prompt_input"  # Sync with session state
)

negative_prompt = st.sidebar.text_area(
    "Negative Prompt",
    value="blurry, low quality, distorted, deformed, ugly",
    height=80,
    help="Describe what you don't want in the image"
)

# Advanced parameters in expander
with st.sidebar.expander("🔧 Advanced Settings"):
    col1, col2 = st.columns(2)
    
    with col1:
        width = st.selectbox("Width", [512, 768, 1024, 1152, 1216], index=2)
        steps = st.slider("Inference Steps", 10, 100, 30, help="More steps = better quality but slower")
    
    with col2:
        height = st.selectbox("Height", [512, 768, 1024, 1152, 1216], index=2)
        guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5, help="How closely to follow the prompt")
    
    # Seed settings
    use_random_seed = st.checkbox("Use Random Seed", value=True)
    seed = None if use_random_seed else st.number_input("Seed", value=42, min_value=0, max_value=999999)

# Generation section
st.sidebar.markdown("---")

# Initialize generation state
if 'generating' not in st.session_state:
    st.session_state.generating = False

generate_button = st.sidebar.button("🚀 Generate Image", type="primary", use_container_width=True, disabled=st.session_state.generating)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎨 Generated Image")
    image_placeholder = st.empty()
    
with col2:
    st.subheader("📊 Generation Info")
    info_placeholder = st.empty()

if generate_button and not st.session_state.generating:
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        st.session_state.generating = True
        
        # Show generation status
        with image_placeholder.container():
            st.info("🎨 Generating your image... Please wait...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress (since we can't get real progress from API)
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Step {i+1}/100")
                time.sleep(0.05)  # Small delay for visual effect
        
        # Generate image
        start_time = time.time()
        result = generate_image(prompt, negative_prompt, width, height, steps, guidance, seed)
        generation_time = time.time() - start_time
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        if result and result.get("success"):
            # Display generation info
            with info_placeholder.container():
                st.success("✅ Generation Complete!")
                st.metric("Generation Time", f"{generation_time:.2f}s")
                st.metric("Resolution", f"{width}×{height}")
                st.metric("Steps", steps)
                st.metric("Guidance", guidance)
                if seed:
                    st.metric("Seed", seed)
            
            # Display generated images
            with image_placeholder.container():
                for i, img_data_url in enumerate(result["images"]):
                    try:
                        # ✅ Just pass the data URL directly to st.image()
                        st.image(img_data_url, caption=f"Generated Image {i+1}", use_column_width=True)

                        # Decode only if you need PIL image (e.g., for download)
                        # Extract base64 part
                        base64_str = img_data_url.split(",")[1]
                        image_data = base64.b64decode(base64_str)
                        image = Image.open(io.BytesIO(image_data))

                        # Add to session state
                        st.session_state.generated_images.append({
                            "image": image,
                            "prompt": prompt,
                            "timestamp": time.time(),
                            "metadata": result["metadata"]
                        })

                        # Download button
                        buf = io.BytesIO()
                        image.save(buf, format="PNG")
                        st.download_button(
                            label=f"📥 Download Image {i+1}",
                            data=buf.getvalue(),
                            file_name=f"sdxl_generated_{int(time.time())}_{i}.png",
                            mime="image/png",
                            key=f"download_current_{i}"
                        )
                        
                        st.success(f"✅ Image {i+1} generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to process image {i+1}: {str(e)}")
            
            # Add to history
            st.session_state.generation_history.append({
                "prompt": prompt,
                "timestamp": time.time(),
                "success": True,
                "processing_time": result["processing_time"]
            })
            
        else:
            with image_placeholder.container():
                error_msg = result.get("message", "Unknown error") if result else "Failed to connect to API"
                st.error(f"❌ Generation failed: {error_msg}")
                
                # Debug information
                with st.expander("🔍 Debug Info"):
                    st.json({
                        "API URL": API_BASE_URL,
                        "Request": {
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "width": width,
                            "height": height,
                            "steps": steps,
                            "guidance": guidance,
                            "seed": seed
                        },
                        "Response": result
                    })
        
        st.session_state.generating = False
        st.rerun()

# Gallery section
if st.session_state.generated_images:
    st.markdown("---")
    st.header("🖼️ Generated Images Gallery")
    
    cols_per_row = 3
    for i in range(0, len(st.session_state.generated_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(st.session_state.generated_images):
                img_data = st.session_state.generated_images[idx]
                with col:
                    st.image(img_data["image"], caption=f"'{img_data['prompt'][:50]}...'", use_column_width=True)
                    
                    buf = io.BytesIO()
                    img_data["image"].save(buf, format="PNG")
                    st.download_button(
                        label="📥 Download",
                        data=buf.getvalue(),
                        file_name=f"gallery_image_{idx}.png",
                        mime="image/png",
                        key=f"download_{idx}_{int(time.time())}"
                    )

# Clear gallery button
if st.session_state.generated_images:
    if st.button("🗑️ Clear Gallery"):
        st.session_state.generated_images = []
        st.rerun()

# Footer with statistics
if st.session_state.generation_history:
    st.markdown("---")
    total_generations = len(st.session_state.generation_history)
    successful_generations = sum(1 for h in st.session_state.generation_history if h["success"])
    avg_time = sum(h.get("processing_time", 0) for h in st.session_state.generation_history) / total_generations
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Generations", total_generations)
    with col2:
        st.metric("Success Rate", f"{(successful_generations/total_generations)*100:.1f}%")
    with col3:
        st.metric("Avg Generation Time", f"{avg_time:.2f}s")

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    1. **Start the API Server**: Make sure your SDXL API server is running (`python main.py`)
    2. **Enter Your Prompt**: Describe what you want to generate in detail
    3. **Adjust Parameters**: Use the sidebar to fine-tune generation settings
    4. **Generate**: Click the generate button and wait for your masterpiece!
    5. **Download**: Save your favorite images using the download buttons
    
    **Tips for Better Results:**
    - Be specific and detailed in your prompts
    - Use negative prompts to avoid unwanted elements
    - Higher steps = better quality but slower generation
    - Guidance scale 7-12 usually works best
    - Try different aspect ratios for variety
    """)

# Sample prompts
with st.expander("💡 Sample Prompts"):
    sample_prompts = [
        "a majestic dragon flying over a medieval castle, fantasy art, highly detailed, 8k",
        "portrait of a cyberpunk character with neon lights, futuristic, digital art",
        "serene Japanese garden with cherry blossoms, traditional architecture, peaceful",
        "abstract geometric patterns in vibrant colors, modern art, minimalist",
        "photorealistic cat sitting on a wooden table, natural lighting, professional photography"
    ]
    
    for prompt_text in sample_prompts:
        if st.button(f"📝 {prompt_text[:50]}...", key=f"sample_{hash(prompt_text)}"):
            st.session_state['prompt_input'] = prompt_text
            st.rerun()