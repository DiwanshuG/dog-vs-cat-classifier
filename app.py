import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Download model from Hugging Face model repo
model_path = hf_hub_download(
    repo_id="mrtaiech/cat-vs-dog-vgg16",  # your model repo
    filename="Best_VGG16_Clean.keras"     # uploaded model filename
)

# Load model
model = load_model(model_path, compile=False)

# Prediction function
def predict(img):
    img = img.convert("RGB")
    img = img.resize((150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = model.predict(x)
    return "Dog 🐶" if preds[0][0] > 0.5 else "Cat 🐱"

# Custom HTML (instead of index.html)
custom_html = """
<div style="text-align:center; padding:20px;">
    <h1 style="color:#4CAF50; font-size:36px;">Dog vs Cat Classifier 🐶🐱</h1>
    <p style="font-size:18px;">Upload an image to find out whether it's a Dog or Cat.</p>
</div>
"""

# Build Gradio UI with HTML + classifier
with gr.Blocks() as demo:
    gr.HTML(custom_html)
    
    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Image", tool="editor")
        output = gr.Label(label="Prediction")
    
    classify_btn = gr.Button("Classify")
    classify_btn.click(fn=predict, inputs=img_input, outputs=output)

if __name__ == "__main__":
    demo.launch()
