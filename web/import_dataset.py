"""Importe un dossier d'images (ex. GHIM-10K) dans le compte d'un utilisateur.

Le dossier devient une catégorie racine et chaque sous-dossier une sous-catégorie.
Usage (l'API doit tourner) : python web/import_dataset.py <dossier> <nom_utilisateur>
"""
import os
import sys

from app import EXTS, add_images, app, db

BATCH = 50

root, username = os.path.abspath(sys.argv[1]), sys.argv[2]
with app.app_context():
    user = db().execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
    if not user:
        sys.exit(f'Utilisateur inconnu : {username} (créez d\'abord le compte sur /register)')
    ids = {}

    def category(rel):
        """Id de la catégorie pour un chemin relatif « a/b/c », créée (avec ses parents) si besoin."""
        if rel not in ids:
            parent = category(os.path.dirname(rel)) if os.path.dirname(rel) else None
            name = os.path.basename(rel)
            row = db().execute('SELECT id FROM categories WHERE user_id = ? AND name = ? AND parent_id IS ?',
                               (user['id'], name, parent)).fetchone()
            ids[rel] = row['id'] if row else db().execute(
                'INSERT INTO categories (user_id, name, parent_id) VALUES (?, ?, ?)', (user['id'], name, parent)).lastrowid
        return ids[rel]

    total = 0
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        files = sorted(f for f in files if os.path.splitext(f)[1].lower() in EXTS)
        if not files:
            continue
        cid = category(os.path.normpath(os.path.join(os.path.basename(root), os.path.relpath(dirpath, root))))
        for i in range(0, len(files), BATCH):
            items = []
            for name in files[i:i + BATCH]:
                with open(os.path.join(dirpath, name), 'rb') as f:
                    items.append((name, f.read()))
            add_images(user['id'], cid, items)
            total += len(items)
            print(f'{total} images importées ({os.path.relpath(dirpath, root)})', flush=True)
    print('Terminé.')
