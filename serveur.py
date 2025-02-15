import os
from flask import Flask
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Configuration pour éviter la surveillance des fichiers supplémentaires
os.environ["FLASK_RUN_EXTRA_FILES"] = ""

from flask import request, jsonify
import os
import cv2
import torch
from ultralytics import YOLO
import librosa
import numpy as np
from model import LSTMClassifier
from flask_cors import CORS

# Configuration
app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les origines par défaut
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
yolo_model = YOLO('best.pt')  # Remplacez 'best.pt' par le chemin correct de votre modèle YOLO
sound_model = torch.load("sound_model1.pth", map_location=torch.device('cpu'))
sound_model.eval()

def predict_sound(audio_path):
    """
    Prédit la classe (cat ou dog) à partir de l'audio enregistré.
    """
    print("je suis bien den predict sound ")
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    max_frames = 60
    if mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]
    else:
        padding = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).T.unsqueeze(0)
    with torch.no_grad():
        output = sound_model(mfcc_tensor)[0]
    confidence = torch.softmax(output, dim=0)
    pred_idx = torch.argmax(output).item()
    pred_class = 'cat' if pred_idx == 1 else 'dog'
    conf_value = confidence[pred_idx].item()
    return pred_class, conf_value

@app.route('/', methods=['POST'])
def process_image_and_sound():
    """
    Traite l'image et l'audio pour détecter les objets correspondant à la classe prédite par l'audio.
    """
    if 'image' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'Both image and audio files are required.'}), 400

    # Récupération des fichiers
    image = request.files['image']
    audio = request.files['audio']

    # Sauvegarde temporaire des fichiers
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    image.save(image_path)
    audio.save(audio_path)
    print("je suis bien arrive la ")

    try:
        # Prédiction de la classe cible à partir de l'audio
        target_class, audio_confidence = predict_sound(audio_path)
        
        print(target_class, audio_confidence)
        # YOLO : Détection d'objets dans l'image
        img = cv2.imread(image_path)
        results = yolo_model.predict(image_path)
        class_names = yolo_model.names

        detections = []
        print("et ce que ....")
        for result in results:
            for box in result.boxes:
                class_index = int(box.cls[0])
                label = class_names[class_index].lower()
                if label == target_class:  # Comparer avec la classe prédite par l'audio
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    if confidence > 0.8:
                        detections.append({
                            'label': label,
                            'confidence': confidence,
                            'box': [x1, y1, x2, y2]
                        })

        return jsonify({
            'audio_prediction': {'class': target_class, 'confidence': audio_confidence},
            'detections': detections
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Nettoyage des fichiers temporaires
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == '__main__':
    app.run(debug=False)
