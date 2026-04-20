# app.py - Streamlit application for Sketch-to-Photo AND Photo-to-Sketch Translation using CycleGAN

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
    """Load both generators from trained CycleGAN model."""
    try:
        # If no model path provided, download from Hugging Face
        if model_path is None:
            model_path = download_model_from_huggingface()
            if model_path is None:
                return None, None, None, None
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Initialize both generators
        generator_sketch_to_photo = Generator(in_channels=3, out_channels=3, ngf=64, n_blocks=6)
        generator_photo_to_sketch = Generator(in_channels=3, out_channels=3, ngf=64, n_blocks=6)
        
        # Load state dicts
        if 'G_AB' in checkpoint and 'G_BA' in checkpoint:
            generator_sketch_to_photo.load_state_dict(checkpoint['G_AB'])
            generator_photo_to_sketch.load_state_dict(checkpoint['G_BA'])
            st.success(" Loaded both generators (Sketch→Photo and Photo→Sketch)")
        elif 'G_AB' in checkpoint:
            generator_sketch_to_photo.load_state_dict(checkpoint['G_AB'])
            generator_photo_to_sketch = None
            st.warning(" Only Sketch→Photo generator available")
        else:
            # Try to load as single generator
            generator_sketch_to_photo.load_state_dict(checkpoint)
            generator_photo_to_sketch = None
            st.warning(" Only one generator found in checkpoint")
        
        generator_sketch_to_photo.eval()
        generator_sketch_to_photo.to(device)
        
        if generator_photo_to_sketch:
            generator_photo_to_sketch.eval()
            generator_photo_to_sketch.to(device)
        
        # Get config and history if available
        config = checkpoint.get('config', {})
        history = checkpoint.get('history', {})
        
        return generator_sketch_to_photo, generator_photo_to_sketch, config, history
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None


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


def translate_image(generator, input_image, device='cpu'):
    """Translate image using the specified generator."""
    with torch.no_grad():
        input_tensor, _ = preprocess_image(input_image)
        input_tensor = input_tensor.to(device)
        
        # Generate output
        output_tensor = generator(input_tensor)
        output_image = postprocess_image(output_tensor)
        
    return output_image


def create_comparison_image(input_img, output_img, input_label="Input", output_label="Output"):
    """Create a side-by-side comparison image."""
    # Ensure both images are same size
    width = max(input_img.width, output_img.width)
    height = max(input_img.height, output_img.height)
    
    input_resized = input_img.resize((width, height), Image.Resampling.LANCZOS)
    output_resized = output_img.resize((width, height), Image.Resampling.LANCZOS)
    
    # Create combined image with labels
    combined = Image.new('RGB', (width * 2, height + 40))
    
    # Paste images
    combined.paste(input_resized, (0, 40))
    combined.paste(output_resized, (width, 40))
    
    # Draw labels (simple text at top)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    try:
        # Try to use default font
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((width//2 - 30, 10), input_label, fill="white", font=font)
    draw.text((width + width//2 - 35, 10), output_label, fill="white", font=font)
    
    return combined


# ========================
# Streamlit UI
# ========================

st.set_page_config(
    page_title="Bidirectional Sketch-Photo Translation",
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
    .direction-selector {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1> Bidirectional Sketch-Photo Translation</h1>
    <p>Transform sketches to photos OR photos to sketches using CycleGAN</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Model Selection")
    
    # Always use Hugging Face model in cloud deployment
    st.info(" Using pre-trained model from Hugging Face")
    st.markdown("Model: [CycleGAN Sketch-Photo](https://huggingface.co/aneelaBashir22f3414/a3_q3_cycle_gan_final)")
    
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
    This application uses **CycleGAN** for bidirectional image translation:
    
    **Two Translation Modes:**
    1. **Sketch → Photo**: Convert hand-drawn sketches to realistic photos
    2. **Photo → Sketch**: Convert real photos to sketch-like drawings
    
    **How it works:**
    1. Choose translation direction
    2. Upload or draw an image
    3. The model translates it
    4. Download the result
    
    **Model Architecture:**
    - Two ResNet-based Generators (6 blocks each)
    - Trained unpaired on sketches and photos
    - 128×128 image size
    """)
    
    st.markdown("---")

# Initialize session state for models
if 'generator_s2p' not in st.session_state:
    st.session_state.generator_s2p = None
    st.session_state.generator_p2s = None
    st.session_state.model_loaded = False

# Load model button or auto-load
if not st.session_state.model_loaded:
    with st.spinner("Loading models from Hugging Face... (First load may take a moment)"):
        gen_s2p, gen_p2s, config, history = load_model(device=device)
        if gen_s2p:
            st.session_state.generator_s2p = gen_s2p
            st.session_state.generator_p2s = gen_p2s
            st.session_state.model_loaded = True
            st.session_state.config = config
            st.session_state.history = history
            st.success(" Models loaded successfully!")

# Main content - Direction Selection
st.markdown("##  Select Translation Direction")

col_dir1, col_dir2 = st.columns(2)

with col_dir1:
    direction = st.radio(
        "Choose translation direction:",
        [" Sketch → Photo", " Photo → Sketch"],
        index=0,
        horizontal=True
    )

# Main content area for input/output
col1, col2 = st.columns(2)

with col1:
    if direction == " Sketch → Photo":
        st.markdown("##  Input Sketch")
        input_label = "Input Sketch"
        output_label = "Generated Photo"
    else:
        st.markdown("##  Input Photo")
        input_label = "Input Photo"
        output_label = "Generated Sketch"
    
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Use Example"],
        horizontal=True
    )
    
    input_image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            f"Upload {input_label.lower()}",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
            help=f"Upload a {input_label.lower()}"
        )
        if uploaded_file:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption=input_label, use_container_width=True)
    
    elif input_method == "Use Example":
        if direction == " Sketch → Photo":
            example_images = {
                "Cat Sketch": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Line_art_cat.svg/512px-Line_art_cat.svg.png",
                "Face Sketch": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Face_sketch.svg/512px-Face_sketch.svg.png",
                "Car Sketch": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Car_sketch.svg/512px-Car_sketch.svg.png",
                "House Sketch": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/House_sketch.svg/512px-House_sketch.svg.png"
            }
        else:
            example_images = {
                "Cat Photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/512px-Cat03.jpg",
                "Flower Photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Flower_poster_2.jpg/512px-Flower_poster_2.jpg",
                "Landscape": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Mount_Hood_reflected_in_Lost_Lake%2C_Oregon.jpg/512px-Mount_Hood_reflected_in_Lost_Lake%2C_Oregon.jpg",
                "Fruit Photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Fruits_%28looking_into_camera%29.jpg/512px-Fruits_%28looking_into_camera%29.jpg"
            }
        
        selected_example = st.selectbox("Choose an example:", list(example_images.keys()))
        
        try:
            response = requests.get(example_images[selected_example], timeout=10)
            input_image = Image.open(io.BytesIO(response.content))
            st.image(input_image, caption=f"Example: {selected_example}", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load example image: {e}")

with col2:
    if direction == " Sketch → Photo":
        st.markdown("## Generated Photo")
    else:
        st.markdown("##  Generated Sketch")
    
    if st.session_state.model_loaded and st.session_state.generator_s2p:
        if input_image is not None:
            # Select the appropriate generator
            if direction == " Sketch → Photo":
                generator = st.session_state.generator_s2p
                direction_text = "sketch to photo"
            else:
                generator = st.session_state.generator_p2s
                direction_text = "photo to sketch"
            
            if generator is None:
                st.error(f" {direction.capitalize()} generator not available in the model checkpoint!")
                st.info("The model checkpoint only contains Sketch→Photo generator. Photo→Sketch translation is not available.")
            else:
                with st.spinner(f"Translating {direction_text}..."):
                    start_time = time.time()
                    output_image = translate_image(generator, input_image, device)
                    inference_time = time.time() - start_time
                
                st.image(output_image, caption=f"{output_label} (took {inference_time:.2f}s)", use_container_width=True)
                
                # Download button
                buf = io.BytesIO()
                output_image.save(buf, format='PNG')
                buf.seek(0)
                st.download_button(
                    label=f" Download {output_label}",
                    data=buf,
                    file_name=f"{direction_text.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
                
                # Show comparison
                st.markdown("---")
                st.markdown("### Side-by-Side Comparison")
                comparison = create_comparison_image(input_image, output_image, input_label, output_label)
                st.image(comparison, caption=f"{input_label} vs {output_label}", use_container_width=True)
                
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
        st.warning("Models are loading. Please wait a moment...")
        st.info("The models will automatically load from Hugging Face. This may take 1-2 minutes on first load.")

# Model Info Section
st.markdown("---")
st.markdown("##  Model Information")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("###  Translation Modes")
    st.markdown("""
    - **Sketch → Photo**: Converts line drawings to realistic images
    - **Photo → Sketch**: Converts real photos to sketch-like drawings
    """)

with col_info2:
    st.markdown("###  Architecture Details")
    st.markdown("""
    - **Generators**: 2 ResNet-based (6 blocks each)
    - **Input Size**: 128×128 pixels
    - **Loss Functions**: Adversarial + Cycle-consistency + Identity
    """)

with col_info3:
    st.markdown("### Tips")
    st.markdown("""
    - Use clear, high-contrast images
    - Center the subject
    - Simple shapes work best
    - Try both directions!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p> Bidirectional Sketch-Photo Translation using CycleGAN | Powered by PyTorch & Hugging Face</p>
    <p style="font-size: 0.8rem;">Convert sketches ↔ photos in both directions! | Model: <a href="https://huggingface.co/aneelaBashir22f3414/a3_q3_cycle_gan_final">Hugging Face Repo</a></p>
</div>
""", unsafe_allow_html=True)
