import gradio as gr
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image
from utils import preprocess_image, postprocess_mask, overlay_mask

# ------------------------------
# CUSTOM METRIC & LOSS
# ------------------------------
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + (1 - dice_coef(y_true, y_pred))

# ------------------------------
# LOAD MODEL
# ------------------------------
model = tf.keras.models.load_model(
    "unet_resnet50v2_final.keras",
    custom_objects={"combined_loss": combined_loss, "dice_coef": dice_coef}
)

# ------------------------------
# SEGMENTATION FUNCTION
# ------------------------------
def segment_ultrasound(image: Image.Image):
    """
    Args:
        image: PIL.Image
    Returns:
        mask (numpy array), overlay (numpy array)
    """
    # Preprocess
    preprocessed = preprocess_image(image)

    # Predict
    pred = model.predict(preprocessed)

    # Postprocess
    mask = postprocess_mask(pred)
    overlay = overlay_mask(image, mask)

    return mask, overlay

# ------------------------------
# GRADIO INTERFACE
# ------------------------------
app = gr.Interface(
    fn=segment_ultrasound,
    inputs=gr.Image(type="pil", label="Upload Ultrasound Image"),
    outputs=[
        gr.Image(label="Segmentation Mask"),
        gr.Image(label="Overlay on Image")
    ],
    title="Ultrasound Segmentation",
    description=(
        "Upload an ultrasound image. "
        "The model will generate a segmentation mask and overlay for visual analysis."
    ),
    examples=None,
)

# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
