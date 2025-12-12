#!/bin/bash

# Script de déploiement sur Hugging Face Spaces

echo "Deploiement de l'API sur Hugging Face Spaces"
echo "=============================================="

# Configuration - Utiliser des variables d'environnement
HF_TOKEN="${HF_TOKEN:-}"  # Définir avec: export HF_TOKEN=your_token
HF_USERNAME="${HF_USERNAME:-votre-username}"  # Définir avec: export HF_USERNAME=your_username
SPACE_NAME="air-traffic-ml-api"

# Vérifier si git est installé
if ! command -v git &> /dev/null; then
    echo "Erreur: git n'est pas installé"
    exit 1
fi

# Vérifier si git-lfs est installé
if ! command -v git-lfs &> /dev/null; then
    echo "Installation de git-lfs..."
    sudo apt-get install git-lfs
    git lfs install
fi

# Créer le dépôt sur Hugging Face (si nécessaire)
echo ""
echo "1. Allez sur https://huggingface.co/new-space"
echo "2. Créez un nouveau Space avec:"
echo "   - Nom: $SPACE_NAME"
echo "   - SDK: Docker"
echo "   - Visibility: Public ou Private"
echo ""
read -p "Appuyez sur Entrée une fois le Space créé..."

# Initialiser le dépôt git local
echo ""
echo "Initialisation du dépôt git..."
cd /home/computer-12/Documents/MODELANAC

if [ ! -d ".git" ]; then
    git init
    git lfs install
fi

# Configurer git LFS pour les fichiers de modèles
git lfs track "*.pkl"
git add .gitattributes

# Ajouter le remote Hugging Face
REMOTE_URL="https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
git remote remove origin 2>/dev/null
git remote add origin $REMOTE_URL

# Copier README.md pour Hugging Face
cp README_HF.md README.md

# Ajouter tous les fichiers
echo ""
echo "Ajout des fichiers..."
git add .
git commit -m "Initial deployment of Air Traffic ML API"

# Push vers Hugging Face
echo ""
echo "Déploiement sur Hugging Face..."
git push -u origin main --force

echo ""
echo "=============================================="
echo "Déploiement terminé!"
echo "Votre API sera disponible sur:"
echo "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo "=============================================="
