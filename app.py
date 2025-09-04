import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model("Best_VGG16_Clean.keras", compile=False)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            return "No file selected!", 400

        # Save uploaded image
        os.makedirs("static", exist_ok=True)
        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        # Preprocess image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        # Predict
        preds = model.predict(x)
        prediction = "Dog 🐶" if preds[0][0] > 0.5 else "Cat 🐱"

    return render_template("index.html", prediction=prediction, image_path=img_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
