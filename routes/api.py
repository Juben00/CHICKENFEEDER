from flask import Blueprint, request, jsonify
from utils.model_utils import get_model, predict_pellets
from utils.model_utils import get_feed_ratio

api_bp = Blueprint('api', __name__, url_prefix='/api')


# New endpoint for pellet counting
@api_bp.route('/count_pellets', methods=['POST'])
def count_pellets():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        model = get_model()
        pellet_count = predict_pellets(model, image)
        return jsonify({'pellet_count': pellet_count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
