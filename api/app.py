"""API REST (Flask + Flask-RESTful) de calcul des descripteurs visuels.

GET  /api/descriptors         -> liste des descripteurs disponibles
POST /api/descriptors         -> tous les descripteurs, pour une ou plusieurs images (champ multipart « images »)
POST /api/descriptors/<nom>   -> un seul descripteur : color_histogram, color_moments, dominant_colors, glcm, lbp
"""
from flask import Flask, request
from flask_restful import Api, Resource, abort
from PIL import Image

from descriptors import DESCRIPTORS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
api = Api(app)


class Descriptors(Resource):
    def get(self, name=None):
        return {'descriptors': list(DESCRIPTORS)}

    def post(self, name=None):
        if name and name not in DESCRIPTORS:
            abort(404, message=f'Descripteur inconnu : {name}. Disponibles : {", ".join(DESCRIPTORS)}')
        files = request.files.getlist('images')
        if not files:
            abort(400, message='Envoyez une ou plusieurs images dans le champ multipart « images ».')
        results = []
        for f in files:
            try:
                img = Image.open(f.stream).convert('RGB')
            except OSError:
                abort(400, message=f'Fichier illisible comme image : {f.filename}')
            if min(img.size) < 3:
                abort(400, message=f'Image trop petite (minimum 3×3 pixels) : {f.filename}')
            results.append({'filename': f.filename, 'width': img.width, 'height': img.height,
                            'descriptors': {n: DESCRIPTORS[n](img) for n in ([name] if name else DESCRIPTORS)}})
        return {'results': results}


api.add_resource(Descriptors, '/api/descriptors', '/api/descriptors/<string:name>')

if __name__ == '__main__':
    app.run(port=5001)
