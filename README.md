Image Processing with Generative AI

This project, "UBDYIK Detiksyon na Mayroong Artipisyal na Katalinuhan", integrates YOLOv8 object detection with LM Studio conversational AI to create a system that not only detects objects in images but also generates meaningful descriptions of them.

📌 Project Overview

The main goal of this project is to enhance image-based insights by combining computer vision and generative AI. Using YOLOv8, the system can accurately detect objects, and with LM Studio, it provides natural language descriptions, making object detection outputs more interactive and user-friendly.

🔹 Key Features

Dataset Preparation: Images are collected, annotated, and processed using Roboflow
.

Model Training: YOLOv8 is trained on Google Colab with custom datasets.

Object Detection: Detects lab-related components such as TV, door, window, chair, table, keyboard, mouse, etc.

Generative AI Integration: LM Studio describes detected objects in natural language.

Streamlit Interface: Provides a simple UI for webcam-based real-time detection and AI-powered descriptions.

🛠️ Tools & Technologies

YOLOv8 (Ultralytics) – Object Detection

Roboflow – Dataset Annotation & Management

Google Colab – Model Training

LM Studio – Generative AI integration

Streamlit – Web-based application

OpenCV & PyTorch – Computer Vision & ML framework

🚀 How It Works

Collect and annotate images via Roboflow.

Train YOLOv8 on Google Colab with custom datasets.

Export the trained model (best.pt) and integrate into the detection pipeline.

Use Streamlit + OpenCV for real-time detection via webcam.

Send detected objects to LM Studio for AI-powered descriptions.

Display both bounding boxes and AI-generated descriptions in the app.

📷 Example Output

YOLO detects objects (TV, table, keyboard, etc.).

LM Studio provides natural descriptions, e.g., “The setup includes a TV on the table with a keyboard and mouse nearby.”

📖 Conclusion

This project demonstrates the power of merging computer vision with conversational AI. YOLO’s accuracy in object detection paired with LM Studio’s natural language generation provides a more accessible and interactive way to interpret complex environments.

Potential applications include:

Education – making learning interactive through AI-powered visual aids

Accessibility – helping visually impaired individuals interpret surroundings

Automation – assisting in smart surveillance and monitoring systems
