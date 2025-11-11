import torch
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import os

# ---------------- Device Setup ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- Load Model ----------------
model = models.mobilenet_v2()
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 2)  # 2 classes: Normal vs Pneumonia

# ⚠️ Make sure to upload mobilenetv2_pneumonia.pth to Colab
if os.path.exists("mobilenetv2_pneumonia.pth"):
    state_dict = torch.load("mobilenetv2_pneumonia.pth", map_location=device)
    model.load_state_dict(state_dict)
else:
    print("⚠️ Model file not found! Please upload mobilenetv2_pneumonia.pth")

model = model.to(device)
model.eval()

# ---------------- Image Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- Prediction Function ----------------
def predict(image):
    if image is None:
        return "Please upload an image", None
    img = image.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        label = "⚠️ Pneumonia Detected" if pred.item() == 1 else "✅ Normal Lungs"
        return label, float(conf.item())

# ---------------- Custom CSS (Dark Mode) ----------------
custom_css = """
body {
  background-color: #0b0c10 !important;
  color: #ffffff !important;
}
.gradio-container {
  background-color: #0b0c10 !important;
  color: #ffffff !important;
}
h1, h2, h3, p, label, span {
  color: #ffffff !important;
}
textarea, input, .output-class, .label {
  background-color: #1f2833 !important;
  color: #ffffff !important;
  border: 1px solid #555555 !important;
}
button {
  background: linear-gradient(90deg, #1f2833, #555555) !important;
  color: #ffffff !important;
  border-radius: 10px !important;
  border: none !important;
  transition: all 0.4s ease !important;
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.1) !important;
}
button:hover {
  background: linear-gradient(90deg, #aaaaaa, #ffffff) !important;
  color: #000000 !important;
  transform: scale(1.05);
  box-shadow: 0 0 20px rgba(255, 255, 255, 0.7) !important;
}
"""

# ---------------- Gradio Interface ----------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📸 Upload Chest X-ray"),
    outputs=[
        gr.Textbox(label="🧠 Prediction"),
        gr.Label(label="📊 Confidence Score")
    ],
    title="🩺 Pneumonia Detection AI",
    description="<b style='color:#ffffff;'>Dark Mode | White Text | AI Diagnosis</b><br>Upload a Chest X-ray to detect Pneumonia using MobileNetV2.",
    theme="default",
    css=custom_css
)

# ---------------- Launch App ----------------
# 'share=True' generates a public link for Colab
interface.launch(share=True)
