from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, time, timedelta
import os
import requests
import json
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.instance_path, 'chickenfeeder.sqlite')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    # Ensure instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register blueprints
    from routes.api import api_bp
    from routes.admin import admin_bp
    app.register_blueprint(api_bp)
    app.register_blueprint(admin_bp)

    # Main dashboard route for root "/"
    @app.route('/')
    def root_dashboard():
        from flask import render_template, redirect, url_for
        from flask_login import current_user
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        # ...fetch stats, logs, schedules...
        return render_template('dashboard.html')
    
    return app

app = create_app()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Initialize scheduler for automated feeding
scheduler = BackgroundScheduler()
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FeedSchedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    feed_time = db.Column(db.Time, nullable=False)
    amount_grams = db.Column(db.Integer, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('schedules', lazy=True))

class DispenseLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    amount_grams = db.Column(db.Integer, nullable=False)
    trigger_type = db.Column(db.String(20), nullable=False)  # 'manual' or 'scheduled'
    schedule_id = db.Column(db.Integer, db.ForeignKey('feed_schedule.id'), nullable=True)
    status = db.Column(db.String(20), default='success')  # 'success' or 'failure'
    error_message = db.Column(db.Text, nullable=True)
    triggered_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    schedule = db.relationship('FeedSchedule', backref=db.backref('dispense_logs', lazy=True))
    user = db.relationship('User', backref=db.backref('dispense_logs', lazy=True))

@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

# IoT Communication Functions
def communicate_with_iot_device(amount_grams):
    """
    Communicate with IoT device to dispense feed
    This is a placeholder - implement based on your IoT device protocol
    """
    try:
        # Example HTTP communication (adjust based on your IoT device)
        # iot_device_url = "http://192.168.1.100:8080/dispense"
        # response = requests.post(iot_device_url, json={'amount': amount_grams}, timeout=10)
        
        # For demonstration, we'll simulate success
        print(f"Dispensing {amount_grams}g of feed to IoT device")
        return True, None
    except Exception as e:
        return False, str(e)

def dispense_feed(amount_grams, trigger_type='manual', schedule_id=None, user_id=None):
    """
    Core function to dispense feed and log the action
    """
    success, error_message = communicate_with_iot_device(amount_grams)
    
    # Log the dispense action
    log_entry = DispenseLog(
        amount_grams=amount_grams,
        trigger_type=trigger_type,
        schedule_id=schedule_id,
        status='success' if success else 'failure',
        error_message=error_message,
        triggered_by=user_id
    )
    
    db.session.add(log_entry)
    db.session.commit()
    
    return success, error_message, log_entry.id

def scheduled_feed_task(schedule_id):
    """
    Task executed by scheduler for automatic feeding
    """
    schedule = db.session.get(FeedSchedule, schedule_id)
    if schedule and schedule.is_active:
        success, error_message, log_id = dispense_feed(
            amount_grams=schedule.amount_grams,
            trigger_type='scheduled',
            schedule_id=schedule_id,
            user_id=schedule.created_by
        )
        
        if not success:
            # Here you could implement email/SMS notifications
            print(f"Scheduled feed failed: {error_message}")

# Routes
@app.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Public registration: create a new user account.
    Admin accounts should be created via the admin dashboard.
    """
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not username or not email or not password:
            flash('All fields are required.')
            return redirect(url_for('register'))
        # uniqueness checks
        if User.query.filter_by(username=username).first():
            flash('Username already taken.')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('register'))
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_admin=False
        )
        db.session.add(user)
        db.session.commit()
        flash('Account created. You may now log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

# Helper: require admin
def require_admin():
    if not current_user.is_authenticated or not getattr(current_user, 'is_admin', False):
        flash('Unauthorized: admin access required.')
        return False
    return True

@app.route('/admin')
@login_required
def admin_dashboard():
    if not require_admin():
        return redirect(url_for('dashboard'))
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/dashboard.html', users=users)

@app.route('/admin/create', methods=['GET', 'POST'])
@login_required
def admin_create_user():
    if not require_admin():
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        is_admin = bool(request.form.get('is_admin'))
        if not username or not email or not password:
            flash('Username, email and password are required.')
            return redirect(url_for('admin_create_user'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('admin_create_user'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('admin_create_user'))
        u = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_admin=is_admin
        )
        db.session.add(u)
        db.session.commit()
        flash('User created successfully.')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin/create_user.html')

@app.route('/admin/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
def admin_edit_user(user_id):
    if not require_admin():
        return redirect(url_for('dashboard'))
    user = db.session.get(User, user_id)
    if not user:
        flash('User not found.')
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', None)
        is_admin = bool(request.form.get('is_admin'))
        # uniqueness checks (exclude this user)
        if username and username != user.username and User.query.filter_by(username=username).first():
            flash('Username already taken.')
            return redirect(url_for('admin_edit_user', user_id=user_id))
        if email and email != user.email and User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('admin_edit_user', user_id=user_id))
        if username:
            user.username = username
        if email:
            user.email = email
        user.is_admin = is_admin
        if password:
            user.password_hash = generate_password_hash(password)
        db.session.commit()
        flash('User updated.')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin/edit_user.html', user=user)

@app.route('/admin/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if not require_admin():
        return redirect(url_for('dashboard'))
    user = db.session.get(User, user_id)
    if not user:
        flash('User not found.')
        return redirect(url_for('admin_dashboard'))
    # Prevent deleting yourself
    if user.id == current_user.id:
        flash('You cannot delete your own account.')
        return redirect(url_for('admin_dashboard'))
    # Ensure at least one admin remains
    if user.is_admin:
        other_admins = User.query.filter(User.is_admin == True, User.id != user.id).count()
        if other_admins == 0:
            flash('Cannot delete the last admin user.')
            return redirect(url_for('admin_dashboard'))
    try:
        db.session.delete(user)
        db.session.commit()
        flash('User deleted.')
    except Exception as e:
        db.session.rollback()
        flash('Error deleting user.')
    return redirect(url_for('admin_dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get today's schedules
    today_schedules = FeedSchedule.query.filter_by(is_active=True).order_by(FeedSchedule.feed_time).all()
    
    # Get today's dispense logs
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_logs = DispenseLog.query.filter(
        DispenseLog.timestamp >= today_start
    ).order_by(DispenseLog.timestamp.desc()).all()
    
    # Calculate total feed dispensed today
    total_today = sum(log.amount_grams for log in today_logs if log.status == 'success')
    
    return render_template('dashboard.html', 
                         schedules=today_schedules,
                         logs=today_logs,
                         total_today=total_today)

@app.route('/schedules')
@login_required
def schedules():
    schedules = FeedSchedule.query.filter_by(created_by=current_user.id).order_by(FeedSchedule.feed_time).all()
    return render_template('schedules.html', schedules=schedules)

@app.route('/schedules/add', methods=['GET', 'POST'])
@login_required
def add_schedule():
    if request.method == 'POST':
        name = request.form['name']
        feed_time_str = request.form['feed_time']
        amount_grams = int(request.form['amount_grams'])
        
        # Parse time
        feed_time = datetime.strptime(feed_time_str, '%H:%M').time()
        
        # --- Limit: 20-150 grams per feeding ---
        if amount_grams < 20 or amount_grams > 150:
            flash('Amount must be between 20 and 150 grams (for 1-5 chickens, 20-30g each).')
            return redirect(url_for('add_schedule'))
        
        schedule = FeedSchedule(
            name=name,
            feed_time=feed_time,
            amount_grams=amount_grams,
            created_by=current_user.id
        )
        
        db.session.add(schedule)
        db.session.commit()
        
        # Add to scheduler
        scheduler.add_job(
            func=scheduled_feed_task,
            trigger=CronTrigger(hour=feed_time.hour, minute=feed_time.minute),
            args=[schedule.id],
            id=f'schedule_{schedule.id}',
            replace_existing=True
        )
        
        flash('Schedule added successfully!')
        return redirect(url_for('schedules'))
    
    return render_template('add_schedule.html')

@app.route('/schedules/<int:schedule_id>/delete', methods=['POST'])
@login_required
def delete_schedule(schedule_id):
    schedule = db.session.get(FeedSchedule, schedule_id)
    if not schedule:
        flash('Schedule not found')
        return redirect(url_for('schedules'))
    if schedule.created_by != current_user.id:
        flash('Unauthorized')
        return redirect(url_for('schedules'))
    
    # Remove from scheduler
    try:
        scheduler.remove_job(f'schedule_{schedule_id}')
    except:
        pass
    
    db.session.delete(schedule)
    db.session.commit()
    
    flash('Schedule deleted successfully!')
    return redirect(url_for('schedules'))

@app.route('/schedules/<int:schedule_id>/toggle', methods=['POST'])
@login_required
def toggle_schedule(schedule_id):
    schedule = db.session.get(FeedSchedule, schedule_id)
    if not schedule:
        return jsonify({'error': 'Schedule not found'}), 404
    if schedule.created_by != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    schedule.is_active = not schedule.is_active
    db.session.commit()
    
    # Update scheduler
    if schedule.is_active:
        scheduler.add_job(
            func=scheduled_feed_task,
            trigger=CronTrigger(hour=schedule.feed_time.hour, minute=schedule.feed_time.minute),
            args=[schedule.id],
            id=f'schedule_{schedule.id}',
            replace_existing=True
        )
    else:
        try:
            scheduler.remove_job(f'schedule_{schedule_id}')
        except:
            pass
    
    return jsonify({'success': True, 'is_active': schedule.is_active})

@app.route('/dispense', methods=['POST'])
@login_required
def manual_dispense():
    """
    API endpoint for manual feed dispensing
    Also serves as IoT integration endpoint
    """
    data = request.get_json()
    amount_grams = data.get('amount', 0)
    
    # --- Limit: 20-150 grams per feeding ---
    if amount_grams < 20 or amount_grams > 150:
        return jsonify({'error': 'Invalid amount. Must be between 20 and 150 grams (for 1-5 chickens, 20-30g each)'}), 400
    
    success, error_message, log_id = dispense_feed(
        amount_grams=amount_grams,
        trigger_type='manual',
        user_id=current_user.id
    )
    
    if success:
        return jsonify({
            'success': True,
            'message': f'Successfully dispensed {amount_grams}g of feed',
            'log_id': log_id
        })
    else:
        return jsonify({
            'success': False,
            'error': error_message
        }), 500

@app.route('/logs')
@login_required
def logs():
    page = request.args.get('page', 1, type=int)
    logs = DispenseLog.query.order_by(DispenseLog.timestamp.desc()).paginate(
        page=page, per_page=50, error_out=False
    )
    return render_template('logs.html', logs=logs)

@app.route('/api/stats')
@login_required
def api_stats():
    """
    API endpoint for dashboard statistics
    """
    # Today's stats
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_logs = DispenseLog.query.filter(DispenseLog.timestamp >= today_start).all()
    
    total_today = sum(log.amount_grams for log in today_logs if log.status == 'success')
    successful_today = len([log for log in today_logs if log.status == 'success'])
    failed_today = len([log for log in today_logs if log.status == 'failure'])
    
    # This week's stats
    week_start = today_start - timedelta(days=7)
    week_logs = DispenseLog.query.filter(DispenseLog.timestamp >= week_start).all()
    total_week = sum(log.amount_grams for log in week_logs if log.status == 'success')
    
    return jsonify({
        'today': {
            'total_grams': total_today,
            'successful_dispenses': successful_today,
            'failed_dispenses': failed_today
        },
        'week': {
            'total_grams': total_week
        }
    })

def create_admin_user():
    """Create default admin user if none exists"""
    if not User.query.first():
        admin = User(
            username='admin',
            email='admin@chickenfeeder.com',
            password_hash=generate_password_hash('admin123'),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()

def setup_scheduled_jobs():
    """Setup all active schedules in the scheduler"""
    active_schedules = FeedSchedule.query.filter_by(is_active=True).all()
    for schedule in active_schedules:
        scheduler.add_job(
            func=scheduled_feed_task,
            trigger=CronTrigger(hour=schedule.feed_time.hour, minute=schedule.feed_time.minute),
            args=[schedule.id],
            id=f'schedule_{schedule.id}',
            replace_existing=True
        )

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def get_feed_ratio():
    if not os.path.exists(CONFIG_PATH):
        return {'pellets': 50, 'grams': 10}
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def set_feed_ratio(pellets, grams):
    with open(CONFIG_PATH, 'w') as f:
        json.dump({'pellets': pellets, 'grams': grams}, f)

@app.route('/admin/feed-ratio', methods=['GET', 'POST'])
@login_required
def admin_feed_ratio():
    if not require_admin():
        return redirect(url_for('dashboard'))
    ratio = get_feed_ratio()
    if request.method == 'POST':
        try:
            pellets = int(request.form.get('pellets', 50))
            grams = float(request.form.get('grams', 10))
            if pellets <= 0 or grams <= 0:
                flash('Values must be positive.', 'danger')
            else:
                set_feed_ratio(pellets, grams)
                flash('Feed-to-gram ratio updated!', 'success')
                return redirect(url_for('admin_feed_ratio'))
        except Exception:
            flash('Invalid input.', 'danger')
    return render_template('admin/feed_ratio.html', ratio=ratio)

# The endpoint that accepts the image to be processed by the model for feed counting is:
# POST /api/predict_feed

# It is defined in routes/api.py as:
# @api_bp.route('/predict_feed', methods=['POST'])
# def predict_feed():
#     # Accepts an image file, runs inference, and returns the predicted pellet count as JSON
#     pass

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_admin_user()
        setup_scheduled_jobs()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
