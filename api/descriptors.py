"""Descripteurs du contenu visuel d'une image (couleur et texture), en NumPy + Pillow.

Auto-vérification : python api/descriptors.py
"""
import numpy as np
from PIL import Image

HIST_BINS = 32        # bins par canal R, G, B
N_COLORS = 5          # nombre de couleurs dominantes
GLCM_LEVELS = 16      # niveaux de gris après quantification
GLCM_OFFSETS = {'0°': (0, 1), '45°': (-1, 1), '90°': (-1, 0), '135°': (-1, -1)}  # (dy, dx), distance 1
LBP_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]  # P=8, R=1


def color_histogram(img):
    """Histogramme normalisé de chaque canal R, G, B."""
    rgb = np.asarray(img)
    return {c: (np.bincount(rgb[..., k].ravel() // (256 // HIST_BINS), minlength=HIST_BINS) / rgb[..., k].size).tolist()
            for k, c in enumerate('rgb')}


def color_moments(img):
    """Moments de couleur (Stricker & Orengo) : moyenne, écart-type et asymétrie de chaque canal."""
    x = np.asarray(img, dtype=np.float64).reshape(-1, 3)
    mean, std = x.mean(0), x.std(0)
    skew = np.cbrt(((x - mean) ** 3).mean(0))
    return {c: {'mean': float(mean[k]), 'std': float(std[k]), 'skewness': float(skew[k])} for k, c in enumerate('rgb')}


def dominant_colors(img):
    """Couleurs dominantes : quantification median-cut affinée par k-means (Pillow), triées par proportion."""
    small = img.copy()
    small.thumbnail((200, 200))
    q = small.quantize(colors=N_COLORS, method=Image.Quantize.MEDIANCUT, kmeans=1)
    palette, total = q.getpalette(), small.width * small.height
    out = []
    for count, idx in sorted(q.getcolors(), reverse=True):
        rgb = palette[3 * idx:3 * idx + 3]
        out.append({'rgb': rgb, 'hex': '#%02x%02x%02x' % tuple(rgb), 'ratio': count / total})
    return out


def glcm(img):
    """Propriétés de Haralick de la matrice de co-occurrence (symétrique, normalisée) pour 4 directions."""
    g = np.asarray(img.convert('L'), dtype=np.int64) * GLCM_LEVELS // 256
    h, w = g.shape
    i, j = np.indices((GLCM_LEVELS, GLCM_LEVELS))
    angles, matrix = {}, None
    for name, (dy, dx) in GLCM_OFFSETS.items():
        r0, r1, c0, c1 = max(0, -dy), h - max(0, dy), max(0, -dx), w - max(0, dx)
        a, b = g[r0:r1, c0:c1], g[r0 + dy:r1 + dy, c0 + dx:c1 + dx]   # paires (pixel, voisin)
        m = np.bincount((a * GLCM_LEVELS + b).ravel(), minlength=GLCM_LEVELS ** 2).reshape(GLCM_LEVELS, GLCM_LEVELS)
        p = (m + m.T) / (2 * m.sum())
        if matrix is None:
            matrix = p                        # P(i, j) pour 0°, renvoyée pour l'affichage
        mu = (i * p).sum()                   # matrice symétrique : mêmes moyenne/variance en i et j
        var = ((i - mu) ** 2 * p).sum()
        asm = (p ** 2).sum()
        angles[name] = {
            'contrast': (p * (i - j) ** 2).sum(),
            'dissimilarity': (p * abs(i - j)).sum(),
            'homogeneity': (p / (1 + (i - j) ** 2)).sum(),
            'energy': np.sqrt(asm),
            'asm': asm,
            'correlation': ((i - mu) * (j - mu) * p).sum() / var if var > 1e-12 else 1.0,
            'entropy': -(p[p > 0] * np.log2(p[p > 0])).sum(),
        }
    props = angles['0°']
    return {'angles': {a: {k: float(v) for k, v in t.items()} for a, t in angles.items()},
            'mean': {k: float(np.mean([t[k] for t in angles.values()])) for k in props},
            'matrix': np.round(matrix, 5).tolist()}


def _transitions(code):
    return sum(((code >> p) & 1) != ((code >> ((p + 1) % 8)) & 1) for p in range(8))


_UNIFORM = [c for c in range(256) if _transitions(c) <= 2]          # 58 motifs uniformes
LBP_MAP = np.full(256, len(_UNIFORM))                                # + 1 bin « non uniformes » = 59 bins
LBP_MAP[_UNIFORM] = np.arange(len(_UNIFORM))


def lbp(img):
    """Histogramme normalisé des motifs binaires locaux uniformes (P=8, R=1) : 59 bins."""
    g = np.asarray(img.convert('L'), dtype=np.int16)
    h, w = g.shape
    c = g[1:-1, 1:-1]
    code = np.zeros(c.shape, dtype=np.uint8)
    for p, (dy, dx) in enumerate(LBP_NEIGHBORS):
        code |= (g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] >= c).astype(np.uint8) << p
    return (np.bincount(LBP_MAP[code].ravel(), minlength=len(_UNIFORM) + 1) / c.size).tolist()


DESCRIPTORS = {'color_histogram': color_histogram, 'color_moments': color_moments,
               'dominant_colors': dominant_colors, 'glcm': glcm, 'lbp': lbp}


if __name__ == '__main__':
    assert len(_UNIFORM) == 58
    flat = Image.new('RGB', (20, 10), (200, 50, 50))
    t = glcm(flat)['mean']
    assert t['contrast'] == 0 and abs(t['energy'] - 1) < 1e-9 and t['homogeneity'] == 1 and t['correlation'] == 1
    assert lbp(flat)[LBP_MAP[255]] == 1                      # tous les voisins >= centre -> motif 11111111
    checker = Image.fromarray((np.indices((10, 10)).sum(0) % 2 * 255).astype(np.uint8)).convert('RGB')
    a = glcm(checker)['angles']
    assert a['0°']['contrast'] == (GLCM_LEVELS - 1) ** 2 and a['45°']['contrast'] == 0
    assert abs(a['0°']['correlation'] + 1) < 1e-9
    h = color_histogram(flat)
    assert all(abs(sum(h[c]) - 1) < 1e-9 for c in 'rgb')
    m = color_moments(flat)['r']
    assert m['mean'] == 200 and m['std'] == 0
    d = dominant_colors(flat)
    assert d[0]['hex'] == '#c83232' and d[0]['ratio'] == 1
    print('OK')
