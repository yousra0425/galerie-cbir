"""Application web de gestion d'une galerie de photos (Flask + SQLite).

Les descripteurs visuels sont calculés par l'API REST (api/app.py). La recherche par similarité et le
retour de pertinence (Query Vector Modification, formule de Rocchio) sont faits ici sur les vecteurs stockés.
"""
import io
import json
import os
import sqlite3
import uuid
import zipfile
from functools import wraps
from pathlib import Path

import numpy as np
import requests
from flask import Flask, abort, flash, g, redirect, render_template, request, send_file, session, url_for
from PIL import Image, ImageEnhance, ImageOps
from werkzeug.security import check_password_hash, generate_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, 'uploads')
DB_PATH = os.path.join(BASE, 'gallery.db')
API_URL = os.environ.get('API_URL', 'http://127.0.0.1:5001/api/descriptors')
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'}
PER_PAGE = 60
ALPHA, BETA, GAMMA = 1.0, 0.75, 0.25   # Rocchio : valeurs par défaut (réglables sur la page de recherche)

# Blocs du vecteur de recherche (suivent les paramètres de api/descriptors.py) :
# histogramme 3×32, moments 3×3, couleurs dominantes 5×(R, G, B, proportion), GLCM 7 propriétés, LBP 59 bins.
BLOCKS = (('hist', 96), ('mom', 9), ('dom', 20), ('glcm', 7), ('lbp', 59))
DESCS = {'hist': ('Histogrammes de couleur', 'Couleur'), 'mom': ('Moments de couleur', 'Couleur'),
         'dom': ('Couleurs dominantes', 'Couleur'), 'glcm': ('GLCM', 'Texture'), 'lbp': ('LBP', 'Texture')}
# Après normalisation z-score, chaque bloc pèse autant dans la distance, quelle que soit sa dimension.
WEIGHTS = np.concatenate([np.full(n, 1 / np.sqrt(n)) for _, n in BLOCKS])
DISTANCES = {'euclid': 'euclidienne', 'l1': 'Manhattan (L1)', 'cos': 'cosinus'}
SORTS = {'recent': ('Date d\'ajout (récent)', 'id DESC'), 'name': ('Nom', 'name COLLATE NOCASE, id'),
         'size': ('Taille (pixels)', 'width * height DESC, id DESC')}
RESAMPLE = {'bilinear': Image.Resampling.BILINEAR, 'nearest': Image.Resampling.NEAREST,
            'bicubic': Image.Resampling.BICUBIC}
# Motifs LBP uniformes (P=8) dans l'ordre des 58 premiers bins renvoyés par l'API.
LBP_UNIFORM = [c for c in range(256) if sum(((c >> p) & 1) != ((c >> ((p + 1) % 8)) & 1) for p in range(8)) <= 2]

SCHEMA = '''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users ON DELETE CASCADE,
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES categories ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users ON DELETE CASCADE,
    category_id INTEGER REFERENCES categories ON DELETE SET NULL,
    name TEXT NOT NULL, file TEXT NOT NULL, width INTEGER, height INTEGER,
    descriptors TEXT NOT NULL,   -- JSON renvoyé par l'API (affichage)
    vector BLOB NOT NULL);       -- float32 concaténés (recherche)
CREATE INDEX IF NOT EXISTS images_user ON images (user_id);
CREATE INDEX IF NOT EXISTS images_category ON images (category_id);
'''

os.makedirs(UPLOADS, exist_ok=True)
key_file = os.path.join(BASE, '.secret_key')
if not os.path.exists(key_file):
    with open(key_file, 'wb') as f:
        f.write(os.urandom(32))
with open(key_file, 'rb') as f:
    secret_key = f.read()

app = Flask(__name__)
app.config.update(SECRET_KEY=secret_key, MAX_CONTENT_LENGTH=500 * 1024 * 1024, SESSION_COOKIE_SAMESITE='Lax')


def db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA foreign_keys = ON')
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    if 'db' in g:
        g.pop('db').close()


with app.app_context():
    db().executescript(SCHEMA)


@app.template_filter()
def num(v, digits=2):
    """Nombre au format français : 186,40 ; −0,84."""
    return f'{v:.{digits}f}'.replace('.', ',').replace('-', '−')


@app.template_filter()
def ko(n):
    return f'{n / 1024:.0f} Ko' if n < 1024 * 1024 else num(n / 1024 / 1024, 1) + ' Mo'


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'uid' not in session:
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped


def own_image(image_id):
    row = db().execute('SELECT * FROM images WHERE id = ? AND user_id = ?', (image_id, session['uid'])).fetchone()
    return row or abort(404)


def own_category(cid):
    """None ou 0 = « non classée » ; sinon la catégorie doit appartenir à l'utilisateur connecté."""
    if not cid:
        return None
    if not db().execute('SELECT 1 FROM categories WHERE id = ? AND user_id = ?', (cid, session['uid'])).fetchone():
        abort(404)
    return cid


def category_tree(uid):
    """Catégories à plat dans l'ordre de l'arbre : [(catégorie, profondeur, nb d'images du sous-arbre)]."""
    rows = db().execute('SELECT c.*, (SELECT COUNT(*) FROM images i WHERE i.category_id = c.id) AS n '
                        'FROM categories c WHERE c.user_id = ? ORDER BY c.name', (uid,)).fetchall()
    children = {}
    for r in rows:
        children.setdefault(r['parent_id'], []).append(r)
    out = []

    def walk(parent, depth):
        total = 0
        for r in children.get(parent, []):
            out.append([r, depth, 0])
            entry = out[-1]
            entry[2] = r['n'] + walk(r['id'], depth + 1)
            total += entry[2]
        return total
    walk(None, 0)
    return out


def cat_path(tree, cid):
    """Catégories de la racine jusqu'à cid (fil d'Ariane)."""
    by_id = {c['id']: c for c, _, _ in tree}
    path, c = [], by_id.get(cid)
    while c:
        path.insert(0, c)
        c = by_id.get(c['parent_id'])
    return path


def lbp_classes(h):
    """Regroupe l'histogramme LBP uniforme (59 bins) en 10 classes invariantes en rotation :
    classe k = motifs uniformes ayant k bits à 1 (k = 0 à 8), classe 9 = motifs non uniformes."""
    out = [0.0] * 10
    for code, v in zip(LBP_UNIFORM, h):
        out[bin(code).count('1')] += v
    out[9] += h[-1]
    return out


class ApiError(Exception):
    pass


def compute_descriptors(items):
    """Appelle l'API REST pour un lot d'images [(nom, octets)] ; renvoie un résultat par image."""
    try:
        r = requests.post(API_URL, files=[('images', (name, data)) for name, data in items], timeout=600)
    except requests.RequestException:
        raise ApiError(f'API des descripteurs injoignable ({API_URL}). Lancez : python api/app.py')
    if not r.ok:
        try:
            msg = r.json()['message']
        except (ValueError, KeyError):
            msg = r.text
        raise ApiError(f'Erreur de l\'API : {msg}')
    return r.json()['results']


def to_vector(d):
    """Concatène les descripteurs renvoyés par l'API en un vecteur (blocs de tailles BLOCKS)."""
    dom = [v for c in d['dominant_colors'] for v in (*c['rgb'], c['ratio'])]
    parts = [d['color_histogram'][c] for c in 'rgb'] + [
        [d['color_moments'][c][m] for c in 'rgb' for m in ('mean', 'std', 'skewness')],
        dom + [0.0] * (dict(BLOCKS)['dom'] - len(dom)),   # moins de 5 couleurs pour les images très uniformes
        list(d['glcm']['mean'].values()),
        d['lbp']]
    v = np.concatenate(parts).astype(np.float32)
    assert v.size == WEIGHTS.size, 'BLOCKS ne correspond plus aux paramètres de l\'API'
    return v


def add_images(uid, cid, items):
    """Enregistre un lot d'images [(nom, octets)] avec leurs descripteurs. Renvoie l'id de la dernière."""
    results = compute_descriptors(items)
    for (name, data), res in zip(items, results):
        file = uuid.uuid4().hex + os.path.splitext(name)[1].lower()
        with open(os.path.join(UPLOADS, file), 'wb') as f:
            f.write(data)
        d = res['descriptors']
        cur = db().execute('INSERT INTO images (user_id, category_id, name, file, width, height, descriptors, vector) '
                           'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                           (uid, cid, name, file, res['width'], res['height'], json.dumps(d), to_vector(d).tobytes()))
    db().commit()
    return cur.lastrowid


# ---------------------------------------------------------------- authentification

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username, password = request.form['username'].strip(), request.form['password']
        if not username or len(password) < 6:
            flash('Nom d\'utilisateur requis et mot de passe d\'au moins 6 caractères.', 'error')
        else:
            try:
                db().execute('INSERT INTO users (username, password) VALUES (?, ?)',
                             (username, generate_password_hash(password)))
                db().commit()
                flash('Compte créé, vous pouvez vous connecter.')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Ce nom d\'utilisateur existe déjà.', 'error')
    return render_template('auth.html', register=True)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = db().execute('SELECT * FROM users WHERE username = ?', (request.form['username'].strip(),)).fetchone()
        if user and check_password_hash(user['password'], request.form['password']):
            session.clear()
            session['uid'], session['username'] = user['id'], user['username']
            session.permanent = bool(request.form.get('remember'))     # « Rester connecté » : 31 jours
            return redirect(url_for('gallery'))
        flash('Identifiants incorrects.', 'error')
    return render_template('auth.html', register=False)


@app.post('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ---------------------------------------------------------------- galerie et catégories

@app.route('/')
@login_required
def gallery():
    uid = session['uid']
    cat = request.args.get('cat', type=int)          # None = toutes, 0 = non classées
    sort = request.args.get('sort') if request.args.get('sort') in SORTS else 'recent'
    page = max(1, request.args.get('page', 1, type=int))
    if cat is None:
        where, params = 'user_id = ?', [uid]
    elif cat == 0:
        where, params = 'user_id = ? AND category_id IS NULL', [uid]
    else:
        own_category(cat)
        where = ('user_id = ? AND category_id IN (WITH RECURSIVE sub(id) AS (SELECT ? UNION ALL '
                 'SELECT c.id FROM categories c JOIN sub ON c.parent_id = sub.id) SELECT id FROM sub)')
        params = [uid, cat]                          # la catégorie et toutes ses sous-catégories
    total = db().execute(f'SELECT COUNT(*) FROM images WHERE {where}', params).fetchone()[0]
    rows = db().execute(f'SELECT id, name, file, width, height FROM images WHERE {where} '
                        f'ORDER BY {SORTS[sort][1]} LIMIT ? OFFSET ?',
                        params + [PER_PAGE, (page - 1) * PER_PAGE]).fetchall()
    images = [dict(r, size=os.path.getsize(os.path.join(UPLOADS, r['file']))) for r in rows]
    tree = category_tree(uid)
    path = cat_path(tree, cat)
    counts = db().execute('SELECT COUNT(*), SUM(category_id IS NULL) FROM images WHERE user_id = ?', (uid,)).fetchone()
    pages = max(1, -(-total // PER_PAGE))
    return render_template('gallery.html', images=images, tree=tree, cat=cat, path=path, counts=counts, total=total,
                           title=path[-1]['name'] if path else ('Non classées' if cat == 0 else 'Toutes les images'),
                           sort=sort, sort_q=None if sort == 'recent' else sort, sorts=SORTS, page=page, pages=pages,
                           page_nums=sorted({1, pages, *range(max(1, page - 2), min(pages, page + 2) + 1)}),
                           first=(page - 1) * PER_PAGE + 1, last=min(page * PER_PAGE, total))


@app.post('/upload')
@login_required
def upload():
    cid = own_category(request.form.get('category_id', type=int))
    items = [(os.path.basename(f.filename.replace('\\', '/')), f.read()) for f in request.files.getlist('images')
             if os.path.splitext(f.filename)[1].lower() in EXTS]
    if not items:
        flash('Aucune image valide (formats : ' + ', '.join(sorted(EXTS)) + ').', 'error')
    else:
        try:
            add_images(session['uid'], cid, items)
            flash(f'{len(items)} image(s) chargée(s), descripteurs calculés.')
        except ApiError as e:
            flash(str(e), 'error')
    return redirect(url_for('gallery', cat=cid))


@app.post('/categories')
@login_required
def create_category():
    name = request.form['name'].strip()
    parent = own_category(request.form.get('parent_id', type=int))
    if not name:
        flash('Nom de catégorie requis.', 'error')
        return redirect(url_for('gallery', cat=parent))
    cur = db().execute('INSERT INTO categories (user_id, name, parent_id) VALUES (?, ?, ?)', (session['uid'], name, parent))
    db().commit()
    return redirect(url_for('gallery', cat=cur.lastrowid))


@app.post('/categories/<int:cid>/delete')
@login_required
def delete_category(cid):
    own_category(cid)
    db().execute('DELETE FROM categories WHERE id = ?', (cid,))   # sous-catégories en cascade, images -> non classées
    db().commit()
    flash('Catégorie supprimée ; ses images sont maintenant « non classées ».')
    return redirect(url_for('gallery'))


@app.post('/images/bulk')
@login_required
def bulk():
    """Télécharger (zip), déplacer ou supprimer une ou plusieurs images."""
    ids = request.form.getlist('ids', type=int)
    rows = db().execute(f'SELECT * FROM images WHERE user_id = ? AND id IN ({",".join("?" * len(ids))})',
                        [session['uid'], *ids]).fetchall()
    back = request.form.get('back', '')
    back = back if back.startswith('/') and not back.startswith('//') else url_for('gallery')
    if not rows:
        flash('Aucune image sélectionnée.', 'error')
        return redirect(back)
    action = request.form['action']
    if action == 'download':
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            for r in rows:
                z.write(os.path.join(UPLOADS, r['file']), f"{r['id']}_{r['name']}")
        buf.seek(0)
        return send_file(buf, mimetype='application/zip', as_attachment=True, download_name='images.zip')
    if action == 'move':
        cid = own_category(request.form.get('category_id', type=int))
        db().executemany('UPDATE images SET category_id = ? WHERE id = ?', [(cid, r['id']) for r in rows])
        flash(f'{len(rows)} image(s) déplacée(s).')
    elif action == 'delete':
        db().executemany('DELETE FROM images WHERE id = ?', [(r['id'],) for r in rows])
        for r in rows:
            Path(UPLOADS, r['file']).unlink(missing_ok=True)
        flash(f'{len(rows)} image(s) supprimée(s).')
    db().commit()
    return redirect(back)


# ---------------------------------------------------------------- image : descripteurs, fichier, transformations

@app.route('/image/<int:image_id>')
@login_required
def image(image_id):
    img = own_image(image_id)
    d = json.loads(img['descriptors'])
    ext = os.path.splitext(img['file'])[1]
    return render_template('image.html', img=img, d=d, lbp10=lbp_classes(d['lbp']),
                           path=cat_path(category_tree(session['uid']), img['category_id']),
                           size=os.path.getsize(os.path.join(UPLOADS, img['file'])),
                           fmt=Image.registered_extensions().get(ext, ext[1:].upper()))


@app.route('/image/<int:image_id>/file')
@login_required
def image_file(image_id):
    img = own_image(image_id)
    return send_file(os.path.join(UPLOADS, img['file']), as_attachment='download' in request.args,
                     download_name=img['name'])


@app.route('/image/<int:image_id>/descriptors.json')
@login_required
def export_descriptors(image_id):
    img = own_image(image_id)
    return send_file(io.BytesIO(img['descriptors'].encode()), mimetype='application/json', as_attachment=True,
                     download_name=os.path.splitext(img['name'])[0] + '_descripteurs.json')


@app.post('/image/<int:image_id>/recompute')
@login_required
def recompute(image_id):
    """Recalcule les descripteurs d'une image via l'API (utile après une mise à jour de l'API)."""
    img = own_image(image_id)
    with open(os.path.join(UPLOADS, img['file']), 'rb') as f:
        data = f.read()
    try:
        d = compute_descriptors([(img['name'], data)])[0]['descriptors']
    except ApiError as e:
        flash(str(e), 'error')
        return redirect(url_for('image', image_id=image_id))
    db().execute('UPDATE images SET descriptors = ?, vector = ? WHERE id = ?',
                 (json.dumps(d), to_vector(d).tobytes(), image_id))
    db().commit()
    flash('Descripteurs recalculés.')
    return redirect(url_for('image', image_id=image_id))


def open_image(row):
    img = Image.open(os.path.join(UPLOADS, row['file']))
    return img if img.mode in ('RGB', 'RGBA', 'L') else img.convert('RGB')


def apply_steps(img, form):
    """Applique dans l'ordre les étapes du formulaire : champs « op » répétés, chacun suivi de ses paramètres."""
    params = {k: iter(form.getlist(k)) for k in ('x', 'y', 'w', 'h', 'scale', 'resample', 'angle', 'direction',
                                                 'brightness')}
    for op in form.getlist('op'):
        if op == 'crop':
            x, y, w, h = (int(float(next(params[k]))) for k in 'xywh')
            if not (x >= 0 and y >= 0 and w >= 1 and h >= 1 and x + w <= img.width and y + h <= img.height):
                raise ValueError('zone de recadrage hors de l\'image')
            img = img.crop((x, y, x + w, y + h))
        elif op == 'resize':
            s = float(next(params['scale'])) / 100
            if not 0 < s <= 4:
                raise ValueError('échelle hors limites')
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                             RESAMPLE.get(next(params['resample']), Image.Resampling.BILINEAR))
        elif op == 'rotate':
            img = img.rotate(-float(next(params['angle'])), expand=True)          # angle positif = sens horaire
        elif op == 'flip':
            img = ImageOps.mirror(img) if next(params['direction']) == 'h' else ImageOps.flip(img)
        elif op == 'gray':
            img = ImageOps.grayscale(img)
        elif op == 'brightness':
            b = float(next(params['brightness'])) / 100
            if not 0 <= b <= 3:
                raise ValueError('luminosité hors limites')
            img = ImageEnhance.Brightness(img).enhance(b)
        else:
            raise ValueError(f'étape inconnue : {op}')
    return img


@app.route('/image/<int:image_id>/transform', methods=['GET', 'POST'])
@login_required
def transform(image_id):
    """GET : éditeur de transformations. POST : enregistre le résultat comme nouvelle image (l'originale reste)."""
    src = own_image(image_id)
    if request.method == 'GET':
        tree = category_tree(session['uid'])
        return render_template('transform.html', img=src, tree=tree, path=cat_path(tree, src['category_id']),
                               stem=os.path.splitext(src['name'])[0], png=src['file'].endswith('.png'))
    try:
        out = apply_steps(open_image(src), request.form)
    except (ValueError, StopIteration):
        flash('Paramètres de transformation invalides.', 'error')
        return redirect(url_for('transform', image_id=image_id))
    png = request.form.get('format') == 'png'
    buf = io.BytesIO()
    if png:
        out.save(buf, 'PNG')
    else:
        (out if out.mode in ('RGB', 'L') else out.convert('RGB')).save(buf, 'JPEG', quality=92)
    stem = os.path.splitext(os.path.basename(request.form.get('name', '').replace('\\', '/')))[0].strip() or 'image'
    try:
        new_id = add_images(session['uid'], own_category(request.form.get('category_id', type=int)),
                            [(stem + ('.png' if png else '.jpg'), buf.getvalue())])
    except ApiError as e:
        flash(str(e), 'error')
        return redirect(url_for('transform', image_id=image_id))
    flash('Nouvelle image enregistrée et analysée.')
    return redirect(url_for('image', image_id=new_id))


@app.post('/image/<int:image_id>/preview')
@login_required
def preview(image_id):
    """Aperçu PNG du résultat des étapes, sans rien enregistrer ; la taille réelle est dans X-Result-Size."""
    try:
        out = apply_steps(open_image(own_image(image_id)), request.form)
    except (ValueError, StopIteration):
        return 'Paramètres invalides', 400
    size = f'{out.width} × {out.height} px'
    out.thumbnail((1000, 1000))
    buf = io.BytesIO()
    out.save(buf, 'PNG')
    buf.seek(0)
    resp = send_file(buf, mimetype='image/png')
    resp.headers['X-Result-Size'] = size.replace('×', 'x')   # en-tête HTTP : ASCII seulement
    return resp


# ---------------------------------------------------------------- recherche par similarité

def distances(Z, q, kind):
    if kind == 'l1':
        return np.abs(Z - q).sum(1)
    if kind == 'cos':
        return 1 - Z @ q / (np.linalg.norm(Z, axis=1) * np.linalg.norm(q) + 1e-12)
    return np.linalg.norm(Z - q, axis=1)


@app.route('/search')
@login_required
def search_pick():
    images = db().execute('SELECT id, name FROM images WHERE user_id = ? ORDER BY id DESC LIMIT 48',
                          (session['uid'],)).fetchall()
    return render_template('search_pick.html', images=images)


@app.route('/search/<int:image_id>', methods=['GET', 'POST'])
@login_required
def search(image_id):
    """GET : recherche simple. POST : relance (action=run) ou itération de retour de pertinence (action=feedback)."""
    query = own_image(image_id)
    # ponytail: tous les vecteurs sont relus à chaque requête (~8 Mo pour 10 000 images) ; mettre en cache si lent.
    rows = db().execute('SELECT id, name, category_id, vector FROM images WHERE user_id = ? ORDER BY id',
                        (session['uid'],)).fetchall()
    V = np.frombuffer(b''.join(r['vector'] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    Z = (V - V.mean(0)) / (V.std(0) + 1e-8) * WEIGHTS       # z-score par dimension, puis poids par bloc
    pos = {r['id']: i for i, r in enumerate(rows)}
    f = request.form
    mode = 'rf' if f.get('mode') == 'rf' else 'simple'
    k = f.get('k', type=int)
    k = k if k in (12, 24, 48) else 12
    dist = f.get('dist') if f.get('dist') in DISTANCES else 'euclid'
    descs = [key for key in DESCS if key in f.getlist('desc')] or list(DESCS)
    alpha, beta, gamma = (min(max(f.get(n, d, type=float), 0.0), 2.0)
                          for n, d in (('alpha', ALPHA), ('beta', BETA), ('gamma', GAMMA)))
    q, iteration, last = Z[pos[image_id]], 0, None
    if f.get('action') == 'feedback':
        # Rocchio : Q' = α·Q + β·moyenne(pertinentes) − γ·moyenne(non pertinentes)
        rel = [pos[i] for i in f.getlist('rel', type=int) if i in pos]
        non = [pos[i] for i in f.getlist('non', type=int) if i in pos]
        q = (alpha * np.array(json.loads(f['q'])) + (beta * Z[rel].mean(0) if rel else 0)
             - (gamma * Z[non].mean(0) if non else 0))
        iteration, last = f.get('iteration', 0, type=int) + 1, {'r': len(rel), 'n': len(non)}
    mask = np.concatenate([np.full(n, float(key in descs)) for key, n in BLOCKS])   # descripteurs comparés
    d = distances(Z * mask, q * mask, dist)
    results = [(rows[i], float(d[i])) for i in np.argsort(d) if rows[i]['id'] != image_id][:k]
    hits = sum(r['category_id'] == query['category_id'] for r, _ in results) if query['category_id'] else None
    tree = category_tree(session['uid'])
    return render_template('search.html', query=query, results=results, q=json.dumps(q.tolist()), hits=hits,
                           mode=mode, k=k, dist=dist, descs=descs, alpha=alpha, beta=beta, gamma=gamma,
                           iteration=iteration, last=last, DESCS=DESCS, DISTANCES=DISTANCES,
                           qpath=' / '.join(c['name'] for c in cat_path(tree, query['category_id'])),
                           cat_names={c['id']: c['name'] for c, _, _ in tree})


if __name__ == '__main__':
    app.run(port=8000)
