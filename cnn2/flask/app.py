from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
import pyttsx3
import threading
import queue
import time

# Load model, scaler, and alphabet mapping
model = tf.keras.models.load_model("/Users/user/Desktop/cnn2/flask/hand_sign_cnn_model.h5")
scaler = joblib.load("/Users/user/Desktop/cnn2/flask/scaler.pkl")
alphabet_mapping = joblib.load("/Users/user/Desktop/cnn2/flask/alphabet_mapping.pkl")
reverse_mapping = {v: k for k, v in alphabet_mapping.items()}


# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Flask app
app = Flask(__name__)

# Variables for tracking
sentence = []
current_prediction = ""
current_sign = None
sign_start_time = None
no_sign_start_time = None

def generate_frames():
    global sentence, current_prediction, current_sign, sign_start_time, no_sign_start_time

    cap = cv2.VideoCapture(0)
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])

                    # Scale and predict
                    normalized_landmarks = scaler.transform([landmarks])
                    if len(model.input_shape) == 4:
                        normalized_landmarks = normalized_landmarks.reshape(1, 21, 2, 1)
                    
                    prediction = model.predict(normalized_landmarks, verbose=0)
                    predicted_label = np.argmax(prediction)
                    predicted_letter = reverse_mapping[predicted_label]
                    current_prediction = predicted_letter

                    # Track sign for confirmation
                    if current_sign != predicted_letter:
                        current_sign = predicted_letter
                        sign_start_time = time.time()
                    elif time.time() - sign_start_time >= 3:
                        sentence.append(predicted_letter)
                        current_sign = None

                    no_sign_start_time = None

            else:
                current_prediction = "No hand detected"
                if no_sign_start_time is None:
                    no_sign_start_time = time.time()
                elif time.time() - no_sign_start_time >= 4:
                    sentence.append(" ")
                    no_sign_start_time = None

            # Display sentence
            sentence_text = "".join(sentence)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error: {e}")
            cap.release()
            cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    global current_prediction, sentence
    return jsonify({'prediction': current_prediction, 'sentence': ''.join(sentence)})

@app.route('/clear_output', methods=['POST'])
def clear_output():
    global sentence
    sentence = []
    return '', 204

def speak_text(text):
    """Function to speak text using pyttsx3 in a separate thread."""
    try:
        engine = pyttsx3.init()
        print(f"Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during speech synthesis: {e}")

@app.route('/read_out', methods=['POST'])
def read_out():
    global sentence
    sentence_text = ''.join(sentence).strip()
    if sentence_text:
        threading.Thread(target=speak_text, args=(sentence_text,)).start()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)