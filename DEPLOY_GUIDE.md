# Guide de Déploiement sur Hugging Face Spaces

## Prérequis

1. Compte Hugging Face: https://huggingface.co/join
2. Token d'accès Hugging Face (déjà généré)
3. Git et Git LFS installés

## Méthode 1: Déploiement Manuel

### Étape 1: Créer un Space sur Hugging Face

1. Allez sur https://huggingface.co/new-space
2. Remplissez:
   - **Owner**: Votre username
   - **Space name**: `air-traffic-ml-api`
   - **License**: MIT
   - **Select the Space SDK**: Docker
   - **Space hardware**: CPU basic (gratuit) ou CPU upgrade (payant mais plus rapide)
   - **Visibility**: Public ou Private

3. Cliquez sur "Create Space"

### Étape 2: Cloner le Space localement

```bash
cd /home/computer-12/Documents/MODELANAC

# Installer git-lfs si nécessaire
sudo apt-get install git-lfs
git lfs install

# Cloner votre Space (remplacez USERNAME par votre nom d'utilisateur)
git clone https://huggingface.co/spaces/USERNAME/air-traffic-ml-api
cd air-traffic-ml-api
```

### Étape 3: Copier les fichiers du projet

```bash
# Revenir au dossier parent
cd /home/computer-12/Documents/MODELANAC

# Copier tous les fichiers nécessaires
cp -r models/ air-traffic-ml-api/
cp -r api/ air-traffic-ml-api/
cp -r config/ air-traffic-ml-api/
cp -r utils/ air-traffic-ml-api/
cp -r scripts/ air-traffic-ml-api/
cp requirements.txt air-traffic-ml-api/
cp Dockerfile air-traffic-ml-api/
cp README_HF.md air-traffic-ml-api/README.md

cd air-traffic-ml-api
```

### Étape 4: Configurer Git LFS pour les modèles

```bash
# Tracker les fichiers de modèles avec Git LFS
git lfs track "*.pkl"
git add .gitattributes
```

### Étape 5: Pousser vers Hugging Face

```bash
# Configurer vos identifiants Git
git config user.email "votre-email@example.com"
git config user.name "Votre Nom"

# Ajouter tous les fichiers
git add .

# Commit
git commit -m "Deploy Air Traffic ML API"

# Pousser vers Hugging Face
git push
```

Quand demandé:
- **Username**: Votre username Hugging Face
- **Password**: Votre token Hugging Face (obtenu depuis https://huggingface.co/settings/tokens)

### Étape 6: Vérifier le déploiement

1. Allez sur `https://huggingface.co/spaces/USERNAME/air-traffic-ml-api`
2. Le Space va automatiquement construire l'image Docker
3. Après quelques minutes, l'API sera accessible
4. Cliquez sur "Logs" pour voir les logs de construction

## Méthode 2: Utilisation du Script Automatique

```bash
cd /home/computer-12/Documents/MODELANAC

# Rendre le script exécutable
chmod +x deploy_hf.sh

# Éditer le script pour ajouter votre username
nano deploy_hf.sh
# Remplacer "votre-username" par votre vrai username Hugging Face

# Lancer le déploiement
./deploy_hf.sh
```

## Méthode 3: Via l'Interface Web (Plus Simple)

1. **Créer le Space** sur https://huggingface.co/new-space
   - Choisir Docker comme SDK

2. **Uploader les fichiers manuellement**:
   - Cliquez sur "Files" dans votre Space
   - Cliquez sur "Add file" > "Upload files"
   - Glissez-déposez tous les dossiers et fichiers:
     - `models/` (avec les .pkl)
     - `api/`
     - `config/`
     - `utils/`
     - `scripts/`
     - `requirements.txt`
     - `Dockerfile`
     - `README_HF.md` (renommer en README.md)

3. **Attendre la construction**: L'image Docker se construit automatiquement

## Configuration du Dockerfile

Le Dockerfile est déjà configuré pour:
- Utiliser Python 3.10
- Installer toutes les dépendances
- Exposer le port 7860 (requis par Hugging Face)
- Lancer l'API FastAPI avec Uvicorn

## Après le Déploiement

Votre API sera accessible sur:
```
https://USERNAME-air-traffic-ml-api.hf.space
```

Documentation interactive:
```
https://USERNAME-air-traffic-ml-api.hf.space/docs
```

## Test de l'API

```bash
# Test depuis votre machine
curl https://USERNAME-air-traffic-ml-api.hf.space/health

# Prédiction
curl -X POST "https://USERNAME-air-traffic-ml-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "callsign": "AFR1234",
    "vitesse_actuelle": 450,
    "altitude": 6000,
    "distance_piste": 40,
    "temperature": 15,
    "vent_vitesse": 25,
    "visibilite": 8,
    "pluie": 2,
    "compagnie": "AF",
    "retard_historique_compagnie": 12,
    "trafic_approche": 8,
    "occupation_tarmac": 0.7,
    "type_avion": "A320",
    "historique_occupation_avion": 45,
    "type_vol": 1,
    "passagers_estimes": 180,
    "disponibilite_emplacements": 12,
    "occupation_actuelle": 0.75,
    "meteo_score": 4,
    "trafic_entrant": 10,
    "trafic_sortant": 6,
    "priorite_vol": 3,
    "emplacements_futurs_libres": 8
  }'
```

## Problèmes Courants

### Les modèles .pkl sont trop gros
- Utilisez Git LFS: `git lfs track "*.pkl"`
- Ou réentraînez avec moins de données

### Le build Docker échoue
- Vérifiez les logs dans l'onglet "Logs" du Space
- Vérifiez que requirements.txt est correct
- Assurez-vous que le Dockerfile est valide

### L'API ne répond pas
- Vérifiez que le port 7860 est utilisé
- Regardez les logs du container
- Vérifiez que les modèles sont bien chargés

## Mise à Jour de l'API

Pour mettre à jour votre API après modifications:

```bash
cd /home/computer-12/Documents/MODELANAC/air-traffic-ml-api
git add .
git commit -m "Update API"
git push
```

Le Space se reconstruira automatiquement.

## Support

- Documentation Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
- Discord Hugging Face: https://hf.co/join/discord
