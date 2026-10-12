import streamlit as st
import pandas as pd
import joblib

# Configuration de la page de l'application
st.set_page_config(page_title="Détecteur d'Anomalies HDFS", page_icon="🛡️", layout="centered")

st.title("🛡️ Détecteur d'Anomalies de Logs HDFS")
st.write("Cette application utilise un **Arbre de Décision** entraîné sur le dataset HDFS pour identifier si des blocs de logs sont normaux ou anormaux.")

# 1. Chargement du modèle préalablement sauvegardé
@st.cache_resource
def charger_modele():
    return joblib.load('arbre_de_decision_HDFS_v1.pkl')

try:
    modele = charger_modele()
except Exception as e:
    st.error(f"Erreur : Impossible de charger le modèle 'modele_arbre_hdfs.pkl'. Avez-vous bien exécuté le code de sauvegarde ? Détails : {e}")
    st.stop()

# 2. Zone de téléchargement du fichier CSV par l'utilisateur
st.header("1. Charger vos données de test")
fichier_uploade = st.file_uploader("Choisissez un fichier CSV (contenant les colonnes d'événements E1 à E29)", type=["csv"])

if fichier_uploade is not None:
    # Lecture du CSV
    df_user = pd.read_csv(fichier_uploade)
    
    st.success("Fichier chargé avec succès !")
    st.write("Aperçu des données chargées :")
    st.dataframe(df_user.head())
    
    # Bouton pour lancer la prédiction
    if st.button("Lancer l'analyse et la prédiction"):
        try:
            # Sélection des colonnes d'événements (E1 à E29)
            colonnes_evenements = [f'E{i}' for i in range(1, 30)]
            
            # Vérification si les colonnes existent dans le fichier
            colonnes_manquantes = [col for col in colonnes_evenements if col not in df_user.columns]
            if colonnes_manquantes:
                st.error(f"Il manque des colonnes d'événements dans votre fichier : {colonnes_manquantes}")
            else:
                X_user = df_user[colonnes_evenements]
                
                # Prédiction avec le modèle
                predictions = modele.predict(X_user)
                
                # Traduction des résultats
                resultats = ["Anomalie ⚠️" if p == 1 else "Normal ✅" for p in predictions]
                
                # Création d'un DataFrame de résultats
                df_resultats = df_user[['BlockId']].copy() if 'BlockId' in df_user.columns else pd.DataFrame()
                df_resultats['Diagnostic'] = resultats
                
                st.header("2. Résultats de la détection")
                st.dataframe(df_resultats)
                
                # Statistiques rapides
                nb_anomalies = resultats.count("Anomalie ⚠️")
                nb_normaux = resultats.count("Normal ✅")
                
                col1, col2 = st.columns(2)
                col1.metric("Blocs Normaux", nb_normaux)
                col2.metric("Blocs Anormaux détectés", nb_anomalies)
                
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction : {e}")