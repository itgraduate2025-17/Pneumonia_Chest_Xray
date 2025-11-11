# Pneumonia_Chest_Xray
## Overview
This project uses a **MobileNetV2** deep learning model built with **PyTorch** to detect **Pneumonia** from **Chest X-ray images**.  
It includes:



## 📂 Project Structure
Pneumonia-Detection/
│
├── Pneumonia_Chest_Xray/ # Dataset folder (NORMAL / PNEUMONIA)
├── training_script.py # Model training script
├── mobilenetv2_pneumonia.pth # Trained model weights
├── app.py # Gragio for UI
└── README.md # Documentation


## ⚙️ Requirements
Install the required packages:

pip install torch torchvision gradio tqdm pillow

## Training the model
python train_mobilenet_pneumonia.py



## 🌐 Running the Flask App
python app.py

## Then open in your browser:
http://127.0.0.1:5000/


## 📈 Results

Final Model Accuracy: 97.8%

📜 License

This project is open-source and available for research and educational purposes.

👨‍💻 Developer

Amjad Ali
