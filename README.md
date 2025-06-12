# 🤟 ASL - American Sign Language Recognition

This project uses a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) alphabets from images. It is designed to help bridge communication gaps for individuals who use sign language by converting hand gestures into readable text.

---

## 📌 Features

- 🔠 Classifies ASL alphabets using CNN  
- 📊 Visualizes training and validation accuracy/loss  
- 🧠 Trained on ASL alphabet dataset  
- 📝 Implemented in Jupyter Notebook using Keras & TensorFlow  

---

## 🧱 Project Structure

ASL/
├── cnn2.ipynb          # Main Jupyter notebook for training and testing the model  
├── README.md           # Project documentation  

---


## 🧠 Model Architecture

The CNN model includes:  
- 3 Convolutional layers  
- MaxPooling layers  
- Dropout for regularization  
- Dense layers for classification  

Activation functions: ReLU for hidden layers, Softmax for output  
Loss function: Categorical Crossentropy  
Optimizer: Adam  

---

## 🏃‍♂️ How to Run

1. Clone this repository:  
   git clone https://github.com/Rishvy/ASL.git  
   cd ASL  

2. (Optional) Install dependencies:  
   pip install -r requirements.txt  

3. Download the dataset and update the path in the notebook (`cnn2.ipynb`).  

4. Open the notebook:  
   jupyter notebook cnn2.ipynb  

5. Run all cells to train and evaluate the model.  

---

## 📈 Training & Evaluation

- Visualizes:  
  - Training vs Validation Accuracy  
  - Training vs Validation Loss  
- Includes confusion matrix to evaluate performance per class  

---

## 🔍 Results

- Achieves high accuracy (up to ~98% depending on dataset split)  
- Handles multiple ASL gestures with high precision  

