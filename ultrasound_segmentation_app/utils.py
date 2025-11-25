import numpy as np
from PIL import Image
import cv2

IMG_SIZE = 256

def preprocess_image(image):
    """Resize and normalize a PIL image."""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype("float32") / 255.0
    if image.shape[-1] == 4:  # remove alpha channel if present
        image = image[..., :3]
    return np.expand_dims(image, axis=0)  # (1,256,256,3)

def postprocess_mask(mask):
    """Convert model output to RGB mask."""
    mask = mask[0, :, :, 0]  # (256,256)
    mask = (mask > 0.5).astype(np.uint8) * 255  # binary mask
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask_rgb

def overlay_mask(image, mask_rgb):
    """Overlay the mask on the original image."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv = cv2.resize(image_cv, (IMG_SIZE, IMG_SIZE))

    color_mask = mask_rgb.copy()
    color_mask[:, :, 0] = 0   # remove blue
    color_mask[:, :, 2] = 255 # red

    overlay = cv2.addWeighted(image_cv, 0.6, color_mask, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
