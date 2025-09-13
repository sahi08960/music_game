from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa
import joblib
import os

app = Flask(__name__)
# Allow cross-origin requests from your React app
CORS(app)

# Define the folder where uploaded files will be temporarily stored
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load the Trained Model and Scaler ---
model = tf.keras.models.load_model('music_genre_model.h5')
scaler = joblib.load('scaler.pkl')

# Define the genres in the correct order
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# --- Audio Preprocessing Function ---
def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict_genre():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            features = extract_features(file_path)
            scaled_features = scaler.transform(features)
            predictions = model.predict(scaled_features)
            predicted_genre_index = np.argmax(predictions[0])
            predicted_genre = genres[predicted_genre_index]
            confidence = float(np.max(predictions[0]))

            return jsonify({
                "predicted_genre": predicted_genre,
                "confidence": confidence
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.remove(file_path)

# --- Root Route (Optional Welcome Message) ---
@app.route('/')
def home():
    return "✅ Music Genre API is running! Use POST /predict to classify audio."

if __name__ == '__main__':
    app.run(debug=True, port=5000)
