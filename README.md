# Pneumonia_Chest_Xray

This project uses a MobileNetV2 deep learning model built with PyTorch to detect Pneumonia from Chest X-ray images.
It is designed for research and educational purposes, demonstrating practical deep learning for medical image classification.

---

## Project Structure

```
Pneumonia-Detection/
│
├── Pneumonia_Chest_Xray/          # Dataset folder containing NORMAL and PNEUMONIA images
├── training_script.py              # Script to train the MobileNetV2 model
├── mobilenetv2_pneumonia.pth      # Pre-trained model weights
├── app.py                          # Gradio-based UI for inference
└── README.md                       # Project documentation
```

---

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision gradio tqdm pillow
```

---

## Training the Model

Run the training script to train MobileNetV2 on your dataset:

```bash
python training_script.py
```

---

## Running the App

Launch the Gradio UI for real-time inference:

```bash
python app.py
```

Then open the following URL in your browser:

```
http://127.0.0.1:5000/
```

---

## Results

* Final Model Accuracy: 97.8%

---

## License

This project is open-source and available for research and educational purposes.

---

## Developer

Amjad Ali
