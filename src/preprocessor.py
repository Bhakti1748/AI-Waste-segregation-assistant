import cv2
import numpy as np
from PIL import Image

def preprocess_image(pil_image: Image.Image, target_size=(640, 640)) -> Image.Image:
    """Preprocess image using OpenCV for better feature recognition."""
    # Convert PIL to OpenCV BGR
    img = np.array(pil_image)
    if img.ndim == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Resize preserving resolution
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 2. Denoise with Bilateral Filter (preserves sharp edges)
    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=50, sigmaSpace=50)

    # Convert back to PIL RGB
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)