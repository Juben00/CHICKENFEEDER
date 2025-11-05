from flask import Blueprint, render_template, request, redirect, url_for, flash
from utils.model_utils import get_feed_ratio, set_feed_ratio

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        pellets = int(request.form.get('pellets', 50))
        grams = float(request.form.get('grams', 10))
        set_feed_ratio(pellets, grams)
        flash('Feed ratio updated!', 'success')
        return redirect(url_for('admin.config'))
    ratio = get_feed_ratio()
    return render_template('admin_config.html', ratio=ratio)
