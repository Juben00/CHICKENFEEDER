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
    if not image or image.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        model = get_model()
        pellet_count = predict_pellets(model, image)
        config = get_feed_ratio()
        pellets = float(config.get('pellets', 1))
        grams = float(config.get('grams', 1))
        if pellets <= 0:
            return jsonify({'error': 'Invalid pellets value in config'}), 500
        grams_to_dispense = round(grams * (pellet_count / pellets), 2)

        # Get user's next active schedule and subtract dispensed grams
        from flask_login import current_user
        from app import db, FeedSchedule
        import datetime
        scheduled_grams = None
        remaining_grams = None
        if current_user.is_authenticated:
            now = datetime.datetime.now().time()
            schedule = db.session.query(FeedSchedule).filter(
                FeedSchedule.created_by == current_user.id,
                FeedSchedule.is_active == True,
                FeedSchedule.feed_time >= now
            ).order_by(FeedSchedule.feed_time.asc()).first()
            if schedule:
                scheduled_grams = schedule.amount_grams
                remaining_grams = round(scheduled_grams - grams_to_dispense, 2)

        return jsonify({
            'pellet_count': pellet_count,
            'grams_to_dispense': grams_to_dispense,
            'scheduled_grams': scheduled_grams,
            'remaining_grams': remaining_grams
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
