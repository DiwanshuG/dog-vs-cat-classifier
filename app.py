import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model_path = hf_hub_download(
    repo_id="mrtaiech/cat-vs-dog-vgg16",  
    filename="Best_VGG16_Clean.keras"     
)

# Load model
model = load_model(model_path, compile=False)


def predict(img):
    img = img.convert("RGB")
    img = img.resize((150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = model.predict(x)
    return "Dog 🐶" if preds[0][0] > 0.5 else "Cat 🐱"


custom_html = """
<div style="text-align:center; padding:20px;">
    <h1 style="color:#4CAF50; font-size:36px;">Dog vs Cat Classifier 🐶🐱</h1>
    <p style="font-size:18px;">Upload an image to find out whether it's a Dog or Cat.</p>
    <br>
    <p style="font-size:16px; color:gray;">
        Created by Diwanshu with ❤️ | Connect with me 
        <a href="https://www.linkedin.com/in/diwanshu-gangwar/" target="_blank" style="color:#0077b5; text-decoration:none; font-weight:bold;">
            LinkedIn
        </a>
    </p>
</div>
"""


with gr.Blocks() as demo:
    gr.HTML(custom_html)
    
    with gr.Row():
        img_input = gr.Image(type="pil", label="Upload Image")
        output = gr.Label(label="Prediction")
    
    classify_btn = gr.Button("Classify")
    classify_btn.click(fn=predict, inputs=img_input, outputs=output)

if __name__ == "__main__":
    demo.launch()
