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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chicken_feeder.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

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
    
    if amount_grams <= 0 or amount_grams > 1000:  # Safety limit
        return jsonify({'error': 'Invalid amount. Must be between 1-1000 grams'}), 400
    
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
        print("Created default admin user: admin/admin123")

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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_admin_user()
        setup_scheduled_jobs()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
