import os
import requests
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
from dotenv import load_dotenv
import logging

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration du logger
logging.basicConfig(level=logging.DEBUG)

# Chemin vers le dossier contenant les images pour l'évaluation
IMAGE_FOLDER = 'static/images'  # Vous pouvez utiliser un chemin relatif ou absolu

# URL de votre API de prédiction
PREDICTION_API_URL = 'https://projet8oc.azurewebsites.net/predict'

@app.route('/')
def index():
    # Charger la liste des images disponibles
    image_list = os.listdir(IMAGE_FOLDER)
    return render_template('index.html', images=image_list)

@app.route('/images', methods=['GET'])
def images():
    # Retourner la liste des images disponibles
    image_list = os.listdir(IMAGE_FOLDER)
    return jsonify(image_list)

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.debug('Requête de prédiction reçue.')
    if 'image_id' not in request.form:
        app.logger.error('Erreur: image_id manquant dans la requête.')
        return jsonify({'error': 'image_id manquant'}), 400

    image_id = request.form['image_id']
    image_path = os.path.join(IMAGE_FOLDER, image_id)
    app.logger.debug(f'Chemin de l\'image : {image_path}')
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            app.logger.debug(f'Envoi du fichier à l\'API de prédiction : {files}')
            response = requests.post(PREDICTION_API_URL, files=files)
            app.logger.debug(f'Réponse brute de l\'API de prédiction : {response.text}')
            response.raise_for_status()
            app.logger.debug(f'Statut de la réponse de l\'API : {response.status_code}')
            try:
                json_response = response.json()
                app.logger.debug(f'Réponse JSON de l\'API : {json_response}')
                mask = json_response.get('mask')
                app.logger.debug(f'Masque prédit : {mask}')
                return jsonify({'image_id': image_id, 'mask': mask})
            except ValueError as json_error:
                app.logger.error(f'Erreur de parsing JSON: {json_error}')
                return jsonify({'error': 'Erreur de parsing JSON', 'response': response.text}), 500
    except requests.exceptions.RequestException as e:
        app.logger.error(f'Erreur lors de l\'appel à l\'API de prédiction: {e}')
        return jsonify({'error': 'Erreur interne du serveur'}), 500
    except Exception as e:
        app.logger.error(f'Erreur inattendue: {e}')
        return jsonify({'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
