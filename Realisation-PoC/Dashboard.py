import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from collections import Counter
import seaborn as sns
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Télécharger les dépendances nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Définir la configuration de la page pour l'accessibilité (Critère 2.4.2 Titre de page)
st.set_page_config(page_title="Analyse de texte et prédiction des allergènes", layout='wide')

# Définir une palette de couleurs à contraste élevé pour l'accessibilité (Critère 1.4.3 Contraste)
sns.set_palette("deep")

# Fonction pour charger les modèles et les ressources avec mise en cache
@st.cache_resource
def load_models_and_resources():
    model_milk = load_model('C:/Users/DELL/Desktop/OpenClass/Formation/Projet_009/Rendu/LSTM_Milk.keras')
    tokenizer_milk = joblib.load('C:/Users/DELL/Desktop/OpenClass/Formation/Projet_009/Rendu/tokenizer_milk.pkl')

    model_others = load_model('C:/Users/DELL/Desktop/OpenClass/Formation/Projet_009/Rendu/LSTM_Other.keras')
    tokenizer_others = joblib.load('C:/Users/DELL/Desktop/OpenClass/Formation/Projet_009/Rendu/tokenizer_others.pkl')
    mlb_others = joblib.load('C:/Users/DELL/Desktop/OpenClass/Formation/Projet_009/Rendu/mlb_others.pkl')

    return model_milk, tokenizer_milk, model_others, tokenizer_others, mlb_others

model_milk, tokenizer_milk, model_others, tokenizer_others, mlb_others = load_models_and_resources()

# Liste des allergènes d'intérêt
allergen_list = ["milk", "nut", "egg", "wheat", "soy", "gluten", "fish", "peanut", "seafood"]

# Initialiser le lemmatizer
lemmatizer = WordNetLemmatizer()

# Fonction de prétraitement pour la recette
def preprocess_recipe(recipe):
    recipe = re.sub(r'[^\w\s]', '', recipe)  # Retirer la ponctuation
    recipe = re.sub(r'\d+', '', recipe)      # Retirer les nombres
    stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
    allergen_set = set(allergen_list)        # Assurer que les allergènes ne sont pas supprimés
    recipe = ' '.join([lemmatizer.lemmatize(word) for word in recipe.lower().split() if word not in stop_words or word in allergen_set])
    return recipe

# Fonction de vérification des mots-clés dans le texte
def check_keywords_in_recipe(recipe, allergen_list):
    found_allergens = [allergen for allergen in allergen_list if allergen in recipe.lower().split()]
    return found_allergens

# Fonction de prédiction pour de nouvelles recettes
def predict_allergens_lstm(recipe):
    # Nettoyage et tokenisation de la recette
    recipe_clean = preprocess_recipe(recipe)
    
    # Première prédiction via le modèle LSTM
    seq = tokenizer.texts_to_sequences([recipe_clean])
    pad_seq = pad_sequences(seq, maxlen=100)
    pred = model.predict(pad_seq)
    
    # Première liste d'allergènes prédits par le modèle LSTM
    lstm_pred_labels = mlb.inverse_transform((pred > 0.25).astype(int))[0]
    
    # Deuxième vérification avec la correspondance des mots clés
    keyword_allergens = check_keywords_in_recipe(recipe_clean, allergen_list)
    
    # Fusionner les deux listes d'allergènes (sans doublons)
    final_allergens = list(set(lstm_pred_labels).union(set(keyword_allergens)))
    
    return final_allergens

# Titre de l'application (Critère 2.4.2 Titre de page)
st.title("Dashboard de détection d'allergènes")

st.write("Utilisez ce Dashboard pour analyser les données textuelles et visualiser les informations clés.")

# Charger les données
uploaded_file = st.file_uploader("Charger un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Aperçu des données :")
    st.write(df.head())

    # Sélectionner la colonne de texte
    column_name = st.selectbox("Sélectionner la colonne contenant le texte", df.columns)
    
    # Sélectionner un allergène
    selected_allergen = st.selectbox("Sélectionner un allergène", ["Aucun"] + allergen_list)
    
    # Fonction pour nettoyer le texte
    def clean_text(text):
        text = re.sub(r"[^\w\s]", '', text)  # Retirer la ponctuation
        text = re.sub(r'\d+', '', text)      # Retirer les nombres
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.lower().split() if word not in stop_words])
        return text

    # Vérification si la colonne sélectionnée contient du texte
    def is_text_column(df, column):
        return pd.api.types.is_string_dtype(df[column])
    
    # Vérifier si la colonne contient du texte
    if is_text_column(df, column_name):
        st.write(f"Analyse de la colonne : {column_name}")

        # Nettoyer et tokeniser le texte
        df['clean_text'] = df[column_name].apply(clean_text)  # Appliquer la fonction de nettoyage
        df['tokens'] = df['clean_text'].apply(nltk.word_tokenize)

        # Filtrer les individus par allergène sélectionné
        if selected_allergen != "Aucun":
            df = df[df['clean_text'].str.contains(selected_allergen, case=False, na=False)]

        # Fréquence des mots
        st.subheader("Fréquence des mots dans la colonne sélectionnée")
        all_words = ' '.join(df['clean_text']).split()
        word_freq = Counter(all_words).most_common(20)
        words, counts = zip(*word_freq)

        # Afficher les fréquences des mots sous forme de tableau
        st.write(pd.DataFrame(word_freq, columns=['Mot', 'Fréquence']))

        # Générer un graphique interactif
        fig, ax = plt.subplots()
        sns.barplot(x=list(words), y=list(counts), ax=ax)
        ax.set_xlabel("Mots")
        ax.set_ylabel("Fréquence")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Taille des phrases (longueur des tokens)
        df['sentence_length'] = df['tokens'].apply(len)
        
        st.subheader("Taille des phrases (Longueur moyenne)")
        st.write("Longueur moyenne des phrases :", df['sentence_length'].mean())

        # Graphique interactif pour la taille des phrases
        st.subheader("Distribution de la longueur des phrases")
        fig, ax = plt.subplots()
        sns.histplot(df['sentence_length'], kde=True, ax=ax)
        ax.set_xlabel("Longueur de la phrase")
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)
        st.caption("Histogramme montrant la distribution de la longueur des phrases.")  # Critère 1.1.1 Contenu non textuel

        # WordCloud
        st.subheader("WordCloud des mots les plus fréquents")
        text = " ".join(all_words)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        st.caption("Nuage de mots représentant les mots les plus fréquents dans le texte.")  # Critère 1.1.1

    else:
        st.error("Cette colonne ne contient pas de texte")

# Interface pour saisir une recette et prédire les allergènes
st.header("Prédiction des allergènes à partir d'une recette")

# Fournir des instructions pour l'accessibilité (par exemple, pour les lecteurs d'écran)
st.write("Veuillez entrer une recette dans la zone de texte ci-dessous. Le modèle prédira les allergènes potentiels.")

# Zone de texte pour saisir la recette (Critère 1.4.4 Redimensionnement du texte)
recipe_input = st.text_area("Entrez la recette ici :", height=200)

if st.button("Prédire les allergènes"):
    if recipe_input:
        predicted_allergens = predict_allergens_lstm(recipe_input)
        if predicted_allergens:
            st.write("**Allergènes potentiels détectés :**")
            st.write(", ".join(predicted_allergens))
        else:
            st.write("Aucun allergène potentiel détecté.")
    else:
        st.write("Veuillez entrer une recette.")

# Ajouter des descriptions textuelles et des étiquettes appropriées (Critère 1.1.1 Contenu non textuel)
st.write("Utilisez ce Dashboard pour analyser les données textuelles et visualiser les informations clés.")

# Assurer que la couleur n'est pas le seul moyen de transmettre de l'information (Critère 1.4.1 Utilisation de la couleur)
# Les graphiques incluent des étiquettes et des légendes pour transmettre l'information sans dépendre uniquement de la couleur.

# Veiller à ce que le contraste entre le texte et l'arrière-plan soit suffisant (Critère 1.4.3 Contraste)
# Les palettes de couleurs utilisées sont conçues pour avoir un contraste élevé.

# Le redimensionnement du texte est géré par le navigateur et Streamlit (Critère 1.4.4 Redimensionnement du texte).