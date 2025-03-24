import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
import pyttsx3

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

# Open webcam
cap = cv2.VideoCapture(1)

print("Real-time hand sign recognition started. Press 'q' to exit and 's' to speak the sentence.")
sentence = []

# Variables for tracking sign consistency
current_sign = None
sign_start_time = None
no_sign_start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Check for hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Normalize landmarks using the scaler
            normalized_landmarks = scaler.transform([landmarks])

            # If using CNN, reshape for input
            if len(model.input_shape) == 4:  # CNN (expects 4D)
                normalized_landmarks = normalized_landmarks.reshape(1, 21, 2, 1)

            # Predict sign
            prediction = model.predict(normalized_landmarks)
            predicted_label = np.argmax(prediction)
            predicted_letter = reverse_mapping[predicted_label]

            # Track sign for 3 seconds for sentence formation
            if predicted_letter == current_sign:
                if (cv2.getTickCount() - sign_start_time) / cv2.getTickFrequency() >= 3:
                    sentence.append(predicted_letter)
                    current_sign = None
            else:
                current_sign = predicted_letter
                sign_start_time = cv2.getTickCount()

            # Display prediction
            cv2.putText(frame, f"Prediction: {predicted_letter}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Reset no sign timer
            no_sign_start_time = None

    else:
        # If no hand is detected, check for 3 seconds for space
        if no_sign_start_time is None:
            no_sign_start_time = cv2.getTickCount()
        elif (cv2.getTickCount() - no_sign_start_time) / cv2.getTickFrequency() >= 4:
            sentence.append(" ")
            print("Added a space.")
            no_sign_start_time = None

    # Display sentence
    sentence_text = "".join(sentence)
    cv2.putText(frame, f"Sentence: {sentence_text}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Speak out the sentence when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("Speaking Sentence:", sentence_text)
        engine.say(sentence_text.strip())
        engine.runAndWait()

    # Show video frame
    cv2.imshow("Hand Sign Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()