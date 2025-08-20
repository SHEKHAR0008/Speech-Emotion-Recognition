# 🗣️ Speech Emotion Recognition (SER) using Machine Learning

## 📌 Project Overview
This project focuses on **recognizing human emotions from speech audio signals** using machine learning and deep learning models.  
We utilize the **RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song)** to train models that can classify different emotional states based on speech patterns.

---

## 📂 Dataset Description
- **Dataset**: [RAVDESS Dataset](https://zenodo.org/record/1188976)  
- The dataset consists of audio-visual recordings with various emotions and intensities.  
- We focus on the **audio-only files** for this project.  

### 🎵 Filename identifiers
Each audio file is structured as:  
`Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav`

For example: `03-02-06-01-02-01-12.wav`
- **03** → Audio-only  
- **02** → Song  
- **06** → Fearful  
- **01** → Normal intensity  
- **02** → Statement “Dogs are sitting by the door”  
- **01** → 1st repetition  
- **12** → Actor ID (even → female, odd → male)  

### 🎭 Emotion Labels
- 01 → Neutral  
- 02 → Calm  
- 03 → Happy  
- 04 → Sad  
- 05 → Angry  
- 06 → Fearful  
- 07 → Disgust  
- 08 → Surprised  

---

## 🤖 Model Description
We implemented and compared **two models**:

### 🔹 Model 1: MLPClassifier (Multi-Layer Perceptron)
- A **feedforward artificial neural network**  
- Maps input speech features (e.g., MFCCs) to emotion labels  
- Simple baseline model for classification  

### 🔹 Model 2: 1D Convolutional Neural Network (CNN)
- Built using **TensorFlow/Keras**  
- Unlike traditional **2D CNNs (Conv2D)** used for images, here we use **1D CNN (Conv1D)** for time-series audio features  
- Layers used:
  - Convolution (Conv1D)  
  - Pooling  
  - Fully Connected (Dense)  
  - Softmax Output for classification  

---

## 📊 Workflow
1. **Data Preprocessing**  
   - Load audio files  
   - Extract features (MFCC, Chroma, Spectral contrast, etc.)  
   - Normalize input features  

2. **Modeling**  
   - Train baseline **MLPClassifier**  
   - Train advanced **1D CNN model** using Keras  

3. **Evaluation**  
   - Compare performance using Accuracy, Precision, Recall, and F1-score  

4. **Experimentation**  
   - Tune hyperparameters  
   - Explore deeper architectures  

