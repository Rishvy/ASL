model = tf.keras.models.load_model("/Users/user/Desktop/cnn2/flask/hand_sign_cnn_model.h5")
scaler = joblib.load("/Users/user/Desktop/cnn2/flask/scaler.pkl")
alphabet_mapping = joblib.load("/Users/user/Desktop/cnn2/flask/alphabet_mapping.pkl")