# Galerie photos : descripteurs visuels et recherche par similarité

Application web (Flask + SQLite) qui consomme une API REST (Flask + Flask-RESTful) de calcul de descripteurs.

## Lancer

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python api/app.py      # terminal 1 : API REST     → http://127.0.0.1:5001/api/descriptors
.venv/bin/python web/app.py      # terminal 2 : application  → http://127.0.0.1:8000
```

Créer un compte sur http://127.0.0.1:8000/register, puis se connecter.

## Importer la base GHIM-10K

Télécharger https://www.kaggle.com/datasets/mcpython/ghim-10k, dézipper, puis (API lancée) :

```bash
.venv/bin/python web/import_dataset.py chemin/vers/GHIM-10K nom_utilisateur
```

Le dossier devient une catégorie racine, chaque sous-dossier une sous-catégorie.

## Structure

| Fichier | Rôle |
|---|---|
| `api/descriptors.py` | Calcul des descripteurs (NumPy + Pillow). `python api/descriptors.py` lance l'auto-vérification. |
| `api/app.py` | API REST Flask-RESTful |
| `web/app.py` | Authentification, galerie, catégories, transformations, recherche, retour de pertinence |
| `web/import_dataset.py` | Import d'un dossier d'images |
| `web/templates/` | Pages HTML (Jinja) |

## API REST

| Méthode | URL | Rôle |
|---|---|---|
| GET | `/api/descriptors` | Liste des descripteurs |
| POST | `/api/descriptors` | Tous les descripteurs pour une ou plusieurs images (champ multipart `images`) |
| POST | `/api/descriptors/<nom>` | Un seul descripteur : `color_histogram`, `color_moments`, `dominant_colors`, `glcm`, `lbp` |

```bash
curl -F images=@photo1.jpg -F images=@photo2.jpg http://127.0.0.1:5001/api/descriptors
```

## Descripteurs

- **Histogrammes de couleur** : 32 bins par canal R, G, B, normalisés.
- **Moments de couleur** : moyenne, écart-type, asymétrie de chaque canal.
- **Couleurs dominantes** : 5 couleurs (median-cut affiné par k-means) avec leur proportion.
- **GLCM** : 16 niveaux de gris, distance 1, directions 0°, 45°, 90°, 135° ; contraste, dissimilarité, homogénéité, énergie, ASM, corrélation, entropie (+ moyenne des 4 directions).
- **LBP** : motifs uniformes P = 8, R = 1, histogramme de 59 bins.

## Recherche

- **Recherche simple** : les descripteurs sont concaténés en un vecteur de 191 valeurs, normalisés (z-score par dimension sur la collection), chaque descripteur pondéré pour peser autant que les autres. On classe les images par distance euclidienne à la requête.
- **Recherche avec retour de pertinence (Query Vector Modification, Rocchio)** : l'utilisateur marque les résultats pertinents (R) et non pertinents (N), puis la requête est déplacée :
  `Q' = α·Q + β·moyenne(R) − γ·moyenne(N)` avec α = 1, β = 0,75, γ = 0,25 (constantes dans `web/app.py`). Chaque itération repart du Q' précédent.
- La **précision** affichée est la part des résultats qui sont dans la même catégorie que la requête (la vérité terrain de GHIM-10K).
