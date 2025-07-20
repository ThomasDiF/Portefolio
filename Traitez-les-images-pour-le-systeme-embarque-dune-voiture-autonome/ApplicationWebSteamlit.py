import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from azure.storage.blob import BlobServiceClient
import io
from dotenv import load_dotenv

# Désactiver l'utilisation du GPU pour TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

# Connexion à Azure Blob Storage
os.environ['AZURE_STORAGE_CONNECTION_STRING'] = 'DefaultEndpointsProtocol=https;AccountName=stockagesave;AccountKey=ehfbE5ikZmNUJ45n9RIRPi95V2FkWmyLYTSn1Bya9mZJlPV8zssPsjucDAxMa4M6Sa/m7t9s0q9a+AStKQP8Og==;EndpointSuffix=core.windows.net'
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not connect_str:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")

container_name = "deeplabv3plusmodel"
blob_name = "Unet_Save.keras"
local_path = "./"
local_file_name = "Unet_Save.keras"

# Télécharger le modèle depuis Azure Blob Storage
@st.cache_resource  # Mettre en cache le modèle après le premier chargement
def load_model_from_azure():
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    download_file_path = os.path.join(local_path, local_file_name)

    # Télécharger le modèle depuis Azure
    if not os.path.exists(download_file_path):
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print("Téléchargement du modèle terminé")
    else:
        print(f"Le fichier modèle existe déjà : {download_file_path}")

    # Charger le modèle localement
    model = load_model(download_file_path, custom_objects={'dice_loss': dice_loss, 'iou': iou})
    return model

# Fonction pour calculer le Dice Loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

# Fonction pour calculer l'IoU corrigé
def iou(y_true, y_pred):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    return tf.reduce_mean((intersection + 1e-6) / (union + 1e-6), axis=0)

# Palette de couleurs pour les masques (adaptée à 8 classes)
def create_jet_palette(num_classes=8):
    cmap = plt.get_cmap('jet', num_classes)
    palette = {i: np.array(cmap(i)[:3]) * 255 for i in range(num_classes)}
    return palette

# Prétraiter l'image
def preprocess_image(image, target_size=(256, 256)):
    original_size = image.size  # Stocker la taille originale
    image = image.resize(target_size, Image.LANCZOS)
    image_np = np.array(image) / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    return image_np, original_size

# Post-traiter le masque prédictif
def postprocess_mask(prediction, original_size):
    mask_resized = np.argmax(prediction, axis=-1)
    mask_resized = np.squeeze(mask_resized)
    mask_img = Image.fromarray(mask_resized.astype(np.uint8), mode='L')
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    return np.array(mask_img)

# Appliquer la palette de couleurs au masque
def color_mask(mask, palette):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in palette.items():
        colored_mask[mask == label] = color
    return colored_mask

# Application Streamlit
st.title('Segmentation d\'images avec U-Net via Azure')

# Charger le modèle depuis Azure Blob Storage
model = load_model_from_azure()

# Télécharger et traiter une image
uploaded_file = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Lire l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image chargée', use_column_width=True)
    st.write("Prédiction en cours...")
    
    # Prétraiter l'image
    processed_image, original_size = preprocess_image(image)
    
    # Faire la prédiction
    prediction = model.predict(processed_image)
    
    # Post-traiter le masque prédictif
    mask_resized = postprocess_mask(prediction, original_size)
    
    # Appliquer la palette de couleurs au masque
    palette = create_jet_palette()
    colored_mask = color_mask(mask_resized, palette)
    
    # Afficher le masque coloré
    st.image(colored_mask, caption='Masque prédictif', use_column_width=True)
