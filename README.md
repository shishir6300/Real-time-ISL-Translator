# 🤟 Real-Time Indian Sign Language (ISL) Recognition System

## 📌 Overview
This project is a real-time Indian Sign Language (ISL) recognition system that uses computer vision and deep learning to detect hand gestures and convert them into text and speech. The system is built using Flask and provides an interactive web interface for real-time gesture recognition and communication assistance.

---

## 🚀 Features

### 🤖 Gesture Recognition
- Real-time hand tracking using MediaPipe  
- Gesture classification using a trained TensorFlow model (`model.h5`)  
- Supports recognition of alphabets and numeric gestures  

### 🔤 Text Formation & Suggestions
- Converts detected gestures into words dynamically  
- Provides intelligent word suggestions using an English dictionary  

### 🔊 Text-to-Speech & Translation
- Converts generated text into speech using Google Text-to-Speech (gTTS)  
- Supports multilingual output with real-time translation using Google Translate API  
- Enables conversion of detected gestures into multiple languages  
- Supports languages such as English, Hindi, Telugu, Tamil, Bengali, Kannada, Malayalam, and Punjabi  

### 🖐️ Multi-Hand Detection
- Detects and processes up to two hands simultaneously  
- Selects nearest hands for accurate gesture recognition  

### 🌐 Web Interface
- Flask-based backend with real-time video streaming  
- Interactive UI for gesture visualization and output display  
- API endpoints for prediction, suggestions, and speech generation  

---

## 🛠️ Tech Stack

- Python  
- Flask  
- TensorFlow / Keras  
- OpenCV  
- MediaPipe  
- NumPy, Pandas  
- gTTS (Text-to-Speech)  
- Google Translate API  
- PyEnchant (Dictionary for suggestions)

---


## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

pip install -r requirements.txt

python main.py

## 📂 Project Structure
