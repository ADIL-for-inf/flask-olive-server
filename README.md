# Flask Olive Server

Ce dépôt contient un serveur Flask léger pour la détection des maladies des feuilles d’olivier, ainsi que le modèle pré-entraîné utilisé pour la prédiction.

## Contenu

- `server.py` : script Flask qui expose l’API pour la détection.
- `models/best.pt` : modèle PyTorch pré-entraîné pour la détection.

## Prérequis

- Python 3.x
- Pip
- Git LFS (pour le modèle `.pt`)

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/ADIL-for-inf/flask-olive-server.git
   cd flask-olive-server
