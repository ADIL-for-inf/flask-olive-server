from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os
import numpy as np
import time
import logging

# Configuration de l'application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paramètres
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Chargement du modèle YOLO
try:
    MODEL_PATH = os.path.join('models', 'best.pt')
    logger.info(f"Chargement du modèle depuis : {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Fichier modèle introuvable : {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    logger.info("Modèle chargé avec succès")
    logger.info(f"Classes disponibles : {model.names}")
    
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
    raise

# Dictionnaire des traitements
TREATMENTS = {
    'Mouche de olivier': {
        'low': "Surveillance recommandée. Pièges à phéromones conseillés.",
        'medium': "Traitement biologique recommandé (spinotoram).",
        'high': "Intervention chimique nécessaire (deltaméthrine)."
    },
    'Tuberculose': {
        'low': "Tailler les branches atteintes.",
        'medium': "Application de bouillie bordelaise.",
        'high': "Traitement fongicide urgent (cuivre + mancozèbe)."
    },
    'cochenille noire': {
        'low': "Lavage des arbres avec de l'eau sous pression.",
        'medium': "Application d'huile horticole.",
        'high': "Traitement avec un insecticide systémique."
    },
    'oeil_de_paon': {
        'low': "Tailler les branches atteintes pour aérer la couronne.",
        'medium': "Application de fongicides à base de cuivre.",
        'high': "Traitement fongicide intensif et élimination des parties très atteintes."
    },
    'psylle': {
        'low': "Surveillance et piégeage.",
        'medium': "Traitement avec des insecticides doux (savon insecticide).",
        'high': "Traitement chimique (diméthoate ou imidaclopride)."
    },
    'en bonne etat': {
        'low': "Continuer les bonnes pratiques culturales.",
        'medium': "Continuer les bonnes pratiques culturales.",
        'high': "Continuer les bonnes pratiques culturales."
    }
}

# Génération des recommandations
def generate_recommendations(disease):
    recommendations = {
        'Mouche de olivier': [
            "Installer des pièges à phéromones",
            "Éliminer les fruits tombés au sol",
            "Traiter tôt le matin"
        ],
        'Tuberculose': [
            "Désinfecter les outils de taille",
            "Éviter l'irrigation par aspersion",
            "Brûler les branches infectées"
        ],
        'cochenille noire': [
            "Favoriser les prédateurs naturels (coccinelles)",
            "Tailler les branches très infestées",
            "Éviter les excès d'engrais azotés"
        ],
        'oeil_de_paon': [
            "Éviter les excès d'humidité",
            "Ramasser et brûler les feuilles tombées",
            "Éclaircir la couronne pour améliorer la circulation d'air"
        ],
        'psylle': [
            "Tailler les pousses atteintes",
            "Éviter les excès d'azote",
            "Utiliser des pièges chromatiques"
        ],
        'en bonne etat': [
            "Continuer les bonnes pratiques culturales",
            "Surveillance mensuelle recommandée"
        ],
        'default': [
            "Isoler la plante affectée",
            "Désinfecter les outils après usage",
            "Consulter un expert agricole"
        ]
    }
    return recommendations.get(disease, recommendations['default'])

# Endpoint principal
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logger.warning("Aucun fichier dans la requête")
        return jsonify({'error': 'Aucune image fournie'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("Fichier vide reçu")
        return jsonify({'error': 'Aucune image sélectionnée'}), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"Format de fichier non supporté : {file.filename}")
        return jsonify({'error': 'Type de fichier non supporté. Formats acceptés: PNG, JPG, JPEG'}), 400
    
    try:
        # Lecture et vérification de l'image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        logger.info(f"Image reçue : {img.size[0]}x{img.size[1]} pixels")
        
        # Prédiction
        start_time = time.time()
        results = model(img)
        inference_time = time.time() - start_time
        
        if not results or len(results[0].boxes) == 0:
            logger.info("Aucun objet détecté dans l'image")
            return jsonify({
                'type': 'no_detection',
                'message': 'Aucune maladie ou feuille détectée',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }), 200
        
        # Traitement des résultats
        result = results[0]
        boxes = result.boxes
        max_conf_idx = boxes.conf.argmax()
        class_id = int(boxes.cls[max_conf_idx])
        confidence = float(boxes.conf[max_conf_idx])
        class_name = result.names[class_id]
        
        logger.info(f"Détection : {class_name} (confiance: {confidence:.2f})")
        logger.info(f"Temps d'inférence : {inference_time:.2f} secondes")
        
        # Détermination du niveau de sévérité
        severity_percent = int(confidence * 100)
        
        # Construction de la réponse pour React Native
        response = {
            'type': 'disease' if class_name != 'en bonne etat' else 'healthy',
            'disease': class_name,
            'severity': severity_percent,
            'confidence': round(confidence * 100, 2),
            'message': f"Détection : {class_name} (confiance: {confidence:.2f})",
            'treatment': TREATMENTS.get(class_name, {}).get('high', "Consulter un spécialiste"),
            'recommendations': generate_recommendations(class_name),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement : {str(e)}")
        return jsonify({
            'error': 'Erreur de traitement',
            'message': str(e)
        }), 500

# Endpoint de test
@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'success',
        'message': 'API opérationnelle',
        'model': 'YOLOv8',
        'classes': model.names,
        'ready': True
    })

# Point d'entrée
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)