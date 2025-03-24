from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
import pyttsx3
import time
import threading

# Load trained model, scaler, and alphabet mapping
model = tf.keras.models.load_model("/Users/user/Desktop/cnn2/flask/hand_sign_cnn_model.h5")
scaler = joblib.load("/Users/user/Desktop/cnn2/flask/scaler.pkl")
alphabet_mapping = joblib.load("/Users/user/Desktop/cnn2/flask/alphabet_mapping.pkl")
reverse_mapping = {v: k for k, v in alphabet_mapping.items()}

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize TTS engine
engine = pyttsx3.init()

# Initialize Flask app
app = Flask(__name__)

# Global variables for state management
sentence = []
current_prediction = ""
current_sign = None
sign_start_time = None
no_sign_start_time = None

def generate_frames():
    """Generator function to process video frames and yield them for streaming."""
    global sentence, current_prediction, current_sign, sign_start_time, no_sign_start_time

    cap = cv2.VideoCapture(1)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract and normalize landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                normalized_landmarks = scaler.transform([landmarks])
                if len(model.input_shape) == 4:  # Reshape for CNN
                    normalized_landmarks = normalized_landmarks.reshape(1, 21, 2, 1)

                # Predict sign
                prediction = model.predict(normalized_landmarks, verbose=0)
                predicted_label = np.argmax(prediction)
                predicted_letter = reverse_mapping[predicted_label]
                current_prediction = predicted_letter

                # Sign consistency logic (3 seconds)
                if current_sign != predicted_letter:
                    current_sign = predicted_letter
                    sign_start_time = time.time()
                elif time.time() - sign_start_time >= 3:
                    sentence.append(predicted_letter)
                    current_sign = None

                no_sign_start_time = None  # Reset no-sign timer

        else:
            current_prediction = "No hand detected"
            if no_sign_start_time is None:
                no_sign_start_time = time.time()
            elif time.time() - no_sign_start_time >= 4:  # 4 seconds no hand = space
                sentence.append(" ")
                no_sign_start_time = None

        # Display sentence on frame
        sentence_text = "".join(sentence)
        cv2.putText(frame, f"Sentence: {sentence_text}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jquery\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream the video feed."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    """Return current prediction and sentence as JSON."""
    global current_prediction, sentence
    return jsonify({
        'prediction': current_prediction,
        'sentence': ''.join(sentence)
    })

@app.route('/clear_output', methods=['POST'])
def clear_output():
    """Clear the sentence."""
    global sentence
    sentence = []
    return '', 204  # No content response

def speak_sentence(text):
    """Speak the given text in a separate thread."""
    engine.say(text)
    engine.runAndWait()

@app.route('/read_out', methods=['POST'])
def read_out():
    """Read out the current sentence in a separate thread."""
    global sentence
    sentence_text = ''.join(sentence).strip()
    if sentence_text:
        threading.Thread(target=speak_sentence, args=(sentence_text,)).start()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)