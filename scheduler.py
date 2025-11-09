from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime


# Check database for logged-in user and their schedule
def feed_chickens():
    from app import app, db, FeedSchedule, User
    with app.app_context():
        # Get all active schedules
        schedules = FeedSchedule.query.filter_by(is_active=True).all()
        for schedule in schedules:
            user = db.session.get(User, schedule.created_by)
            print(f"Feeding triggered for user {user.username} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Scheduled time: {schedule.feed_time}, Amount: {schedule.amount_grams}g")
            # TODO: Add actual feeding logic here (e.g., GPIO, database update, etc.)

scheduler = BackgroundScheduler()
# Example: Feed every day at 8:00 AM
scheduler.add_job(feed_chickens, 'cron', hour=8, minute=0)

# Start the scheduler (call this in app.py)
def start_scheduler():
    if not scheduler.running:
        scheduler.start()
