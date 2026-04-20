# app.py - Streamlit application for Sketch-to-Photo Translation using CycleGAN
# Updated to work with Hugging Face model hosting for Streamlit Cloud

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import os
import tempfile
import time
import requests
from datetime import datetime

# ========================
# Model Architecture (same as training)
# ========================

class ResNetBlock(nn.Module):
    """Residual block with reflection padding and instance normalization."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """ResNet-based Generator for CycleGAN."""
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=6):
        super().__init__()
        assert n_blocks >= 0

        # Initial convolution block: c7s1-64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling: d128, d256
        in_features = ngf
        out_features = ngf * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # ResNet blocks: R256 × n_blocks
        for _ in range(n_blocks):
            model += [ResNetBlock(in_features)]

        # Upsampling: u128, u64
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer: c7s1-3
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ========================
# Helper Functions
# ========================

def download_model_from_huggingface():
    """Download model from Hugging Face with caching."""
    model_url = "https://huggingface.co/aneelaBashir22f3414/a3_q3_cycle_gan_final/resolve/main/cyclegan_final%20.pth"
    
    # Create cache directory
    cache_dir = os.path.join(tempfile.gettempdir(), "cyclegan_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file path
    cache_file = os.path.join(cache_dir, "cyclegan_final.pth")
    
    # Check if model already exists in cache
    if os.path.exists(cache_file):
        st.info("Loading model from cache...")
        return cache_file
    
    # Download model with progress bar
    with st.spinner("Downloading model from Hugging Face... (This may take a few minutes on first run)"):
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            with open(cache_file, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    progress_bar = st.progress(0)
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress_bar.progress(min(1.0, downloaded / total_size))
                    progress_bar.progress(1.0)
            
            st.success("Model downloaded successfully!")
            return cache_file
            
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None


def load_model(model_path=None, device='cpu'):
    """Load trained generator model."""
    try:
        # If no model path provided, download from Hugging Face
        if model_path is None:
            model_path = download_model_from_huggingface()
            if model_path is None:
                return None, None, None
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Initialize model
        generator = Generator(in_channels=3, out_channels=3, ngf=64, n_blocks=6)
        
        # Handle different checkpoint formats
        if 'G_AB' in checkpoint:
            generator.load_state_dict(checkpoint['G_AB'])
        elif 'model_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
        
        generator.eval()
        generator.to(device)
        
        # Get config and history if available
        config = checkpoint.get('config', {})
        history = checkpoint.get('history', {})
        
        return generator, config, history
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def preprocess_image(image, target_size=128):
    """Preprocess image for model input."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize to [-1, 1]
    img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, image


def postprocess_image(tensor):
    """Convert tensor back to PIL image."""
    # Denormalize from [-1, 1] to [0, 1]
    img = tensor.squeeze(0).cpu().detach()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def translate_sketch(generator, sketch_image, device='cpu'):
    """Translate sketch to photo using the generator."""
    with torch.no_grad():
        sketch_tensor, _ = preprocess_image(sketch_image)
        sketch_tensor = sketch_tensor.to(device)
        
        # Generate photo
        photo_tensor = generator(sketch_tensor)
        photo_image = postprocess_image(photo_tensor)
        
    return photo_image


def create_comparison_image(sketch, photo):
    """Create a side-by-side comparison image."""
    # Ensure both images are same size
    width = max(sketch.width, photo.width)
    height = max(sketch.height, photo.height)
    
    sketch_resized = sketch.resize((width, height), Image.Resampling.LANCZOS)
    photo_resized = photo.resize((width, height), Image.Resampling.LANCZOS)
    
    # Create combined image
    combined = Image.new('RGB', (width * 2, height))
    combined.paste(sketch_resized, (0, 0))
    combined.paste(photo_resized, (width, 0))
    
    return combined


# ========================
# Streamlit UI
# ========================

st.set_page_config(
    page_title="Sketch-to-Photo Translation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 0.5rem 0 0 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        color: #004085;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1> Sketch-to-Photo Translation</h1>
    <p>Transform your hand-drawn sketches into realistic photos using CycleGAN</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Model Selection")
    
    # Always use Hugging Face model in cloud deployment
    st.info(" Using pre-trained model from Hugging Face")
    st.markdown("Model: [CycleGAN Sketch-to-Photo](https://huggingface.co/aneelaBashir22f3414/a3_q3_cycle_gan_final)")
    
    st.markdown("---")
    
    st.markdown("## Settings")
    device_option = st.selectbox(
        "Device",
        ["Auto (GPU if available)", "CPU only"],
        index=0
    )
    
    if device_option == "Auto (GPU if available)":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    st.info(f"Using device: **{device.upper()}**")
    
    st.markdown("---")
    
    st.markdown("## About")
    st.markdown("""
    This application uses **CycleGAN** to convert hand-drawn sketches into realistic photos.
    
    **How it works:**
    1. Upload a sketch (or use the drawing canvas)
    2. The model translates it to a photo
    3. Download the result
    
    **Model Architecture:**
    - ResNet-based Generator with 6 blocks
    - Trained on sketches and photos
    - 128×128 image size
    
    **Deployment:**
    - Model hosted on Hugging Face Hub
    - App deployed on Streamlit Cloud
    """)
    
    st.markdown("---")
    
    # Show system info
    if torch.cuda.is_available():
        st.markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")
        st.markdown(f"**VRAM:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Initialize session state for model
if 'generator' not in st.session_state:
    st.session_state.generator = None
    st.session_state.model_loaded = False

# Load model button or auto-load
if not st.session_state.model_loaded:
    with st.spinner("Loading model from Hugging Face... (First load may take a moment)"):
        generator, config, history = load_model(device=device)
        if generator:
            st.session_state.generator = generator
            st.session_state.model_loaded = True
            st.session_state.config = config
            st.session_state.history = history
            st.success(" Model loaded successfully!")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown("## Input Sketch")
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Draw Sketch", "Use Example"]
    )
    
    sketch_image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a sketch image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
            help="Upload a hand-drawn sketch or any image"
        )
        if uploaded_file:
            sketch_image = Image.open(uploaded_file)
            st.image(sketch_image, caption="Uploaded Sketch", use_container_width=True)
    
    elif input_method == "Draw Sketch":
        try:
            from streamlit_drawable_canvas import st_canvas
            st.markdown("Draw your sketch below:")
            drawing = st_canvas(
                stroke_width=3,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=300,
                width=300,
                key="canvas",
                display_toolbar=True
            )
            if drawing.image_data is not None:
                sketch_image = Image.fromarray(drawing.image_data.astype('uint8'), mode='RGBA')
                sketch_image = sketch_image.convert('RGB')
                st.image(sketch_image, caption="Your Drawing", use_container_width=True)
        except ImportError:
            st.warning("Drawing canvas requires streamlit-drawable-canvas. Using file upload instead.")
            uploaded_file = st.file_uploader(
                "Upload a sketch image",
                type=['png', 'jpg', 'jpeg'],
                key="fallback_upload"
            )
            if uploaded_file:
                sketch_image = Image.open(uploaded_file)
                st.image(sketch_image, caption="Uploaded Sketch", use_container_width=True)
    
    elif input_method == "Use Example":
        st.markdown("Select an example sketch:")
        example_images = {
            "Cat Sketch": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Line_art_cat.svg/512px-Line_art_cat.svg.png",
            "Face Sketch": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Face_sketch.svg/512px-Face_sketch.svg.png",
            "Car Sketch": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Car_sketch.svg/512px-Car_sketch.svg.png",
            "House Sketch": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/House_sketch.svg/512px-House_sketch.svg.png"
        }
        
        selected_example = st.selectbox("Choose an example:", list(example_images.keys()))
        
        try:
            response = requests.get(example_images[selected_example], timeout=10)
            sketch_image = Image.open(io.BytesIO(response.content))
            st.image(sketch_image, caption=f"Example: {selected_example}", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load example image: {e}")

with col2:
    st.markdown("## Generated Photo")
    
    if st.session_state.model_loaded and st.session_state.generator:
        if sketch_image is not None:
            with st.spinner("Translating sketch to photo..."):
                start_time = time.time()
                photo_image = translate_sketch(st.session_state.generator, sketch_image, device)
                inference_time = time.time() - start_time
            
            st.image(photo_image, caption=f"Generated Photo (took {inference_time:.2f}s)", use_container_width=True)
            
            # Download button
            buf = io.BytesIO()
            photo_image.save(buf, format='PNG')
            buf.seek(0)
            st.download_button(
                label=" Download Generated Photo",
                data=buf,
                file_name=f"generated_photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
            
            # Show comparison
            st.markdown("---")
            st.markdown("### Side-by-Side Comparison")
            comparison = create_comparison_image(sketch_image, photo_image)
            st.image(comparison, caption="Sketch vs Generated Photo", use_container_width=True)
            
            # Download comparison
            buf_comp = io.BytesIO()
            comparison.save(buf_comp, format='PNG')
            buf_comp.seek(0)
            st.download_button(
                label=" Download Comparison",
                data=buf_comp,
                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
    else:
        st.warning("Model is loading. Please wait a moment...")
        st.info("The model will automatically load from Hugging Face. This may take 1-2 minutes on first load.")

# Additional features section
st.markdown("---")
st.markdown("## Tips & Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Tips for Best Results")
    st.markdown("""
    - Use clear, well-defined sketches
    - Ensure good contrast (black on white)
    - Keep the drawing centered
    - Simple shapes work best
    - Avoid too many details
    """)

with col2:
    st.markdown("###  Model Details")
    if st.session_state.get('history') and st.session_state.history:
        if 'G_loss' in st.session_state.history and st.session_state.history['G_loss']:
            st.markdown(f"**Generator Loss:** {st.session_state.history['G_loss'][-1]:.4f}")
        if 'D_loss' in st.session_state.history and st.session_state.history['D_loss']:
            st.markdown(f"**Discriminator Loss:** {st.session_state.history['D_loss'][-1]:.4f}")
    st.markdown("**Architecture:** ResNet-6 blocks")
    st.markdown("**Input Size:** 128×128 pixels")
    st.markdown("**Training:** CycleGAN with identity loss")

with col3:
    st.markdown("###  Deployment Info")
    st.markdown("""
    - **Model Host:** Hugging Face Hub
    - **App Host:** Streamlit Cloud
    - **Framework:** PyTorch
    - **Status:** Active
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p> Sketch-to-Photo Translation using CycleGAN | Powered by PyTorch & Hugging Face</p>
    <p style="font-size: 0.8rem;">Upload a sketch and let the AI transform it into a photo! | Model: <a href="https://huggingface.co/aneelaBashir22f3414/a3_q3_cycle_gan_final">Hugging Face Repo</a></p>
</div>
""", unsafe_allow_html=True)
