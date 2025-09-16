# Chicken Feed Management System

A comprehensive web-based system for managing automated chicken feeding with IoT device integration, scheduling, and monitoring capabilities.

## Features

### 🔐 Authentication
- Admin login system for farm owners
- Secure session management
- Default admin account (admin/admin123)

### ⏰ Feed Scheduling
- Create multiple feeding schedules per day
- Specify exact times and feed amounts (1-1000g)
- Enable/disable schedules dynamically
- Template-based quick setup

### 🎮 Manual Control
- Instant feed dispensing with custom amounts
- Real-time feedback and status updates
- Safety limits and validation

### 🔌 IoT Integration
- RESTful API endpoint (`/dispense`) for IoT communication
- Configurable device communication (HTTP/MQTT/Socket)
- Automatic retry and error handling

### 📊 Data Logging
- Complete audit trail of all dispense actions
- Success/failure status tracking
- Timestamp and amount logging
- User attribution

### 📈 Dashboard & Analytics
- Real-time statistics and summaries
- Today's schedule overview
- Recent activity monitoring
- Auto-refreshing stats

### 🔔 Notifications
- Failed dispense alerts
- System status notifications
- Real-time UI updates

### 🗄️ Database
- SQLite database (easily upgradeable to MySQL/PostgreSQL)
- Normalized schema for schedules, logs, and users
- Automatic migrations and setup

## Technology Stack

- **Backend**: Flask (Python)
- **Database**: SQLAlchemy with SQLite
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Scheduling**: APScheduler
- **Authentication**: Flask-Login
- **Icons**: Font Awesome

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CHICKENFEEDER
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the system**
   - Open your browser to `http://localhost:5000`
   - Login with default credentials: `admin` / `admin123`

## Configuration

### IoT Device Integration

The system includes a placeholder for IoT device communication in the `communicate_with_iot_device()` function. Customize this based on your device:

#### HTTP Communication Example:
```python
def communicate_with_iot_device(amount_grams):
    try:
        response = requests.post(
            "http://192.168.1.100:8080/dispense",
            json={'amount': amount_grams},
            timeout=10
        )
        return response.status_code == 200, None
    except Exception as e:
        return False, str(e)
```

#### MQTT Communication Example:
```python
import paho.mqtt.client as mqtt

def communicate_with_iot_device(amount_grams):
    try:
        client = mqtt.Client()
        client.connect("mqtt.broker.address", 1883, 60)
        client.publish("chicken/feed/dispense", str(amount_grams))
        return True, None
    except Exception as e:
        return False, str(e)
```

### Database Configuration

By default, the system uses SQLite. To use MySQL or PostgreSQL:

1. Install the appropriate database driver:
   ```bash
   # For MySQL
   pip install PyMySQL
   
   # For PostgreSQL
   pip install psycopg2-binary
   ```

2. Update the database URI in `app.py`:
   ```python
   # MySQL
   app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://user:password@localhost/chickenfeeder'
   
   # PostgreSQL
   app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/chickenfeeder'
   ```

### Security Configuration

**Important**: Change the secret key for production:

```python
app.config['SECRET_KEY'] = 'your-secure-random-secret-key'
```

## API Endpoints

### POST /dispense
Dispense feed manually or from IoT devices.

**Request:**
```json
{
    "amount": 50
}
```

**Response (Success):**
```json
{
    "success": true,
    "message": "Successfully dispensed 50g of feed",
    "log_id": 123
}
```

**Response (Error):**
```json
{
    "success": false,
    "error": "Invalid amount. Must be between 1-1000 grams"
}
```

### GET /api/stats
Get current statistics for dashboard updates.

**Response:**
```json
{
    "today": {
        "total_grams": 250,
        "successful_dispenses": 5,
        "failed_dispenses": 0
    },
    "week": {
        "total_grams": 1750
    }
}
```

## Usage Guide

### Creating Feed Schedules

1. Navigate to **Schedules** → **Add Schedule**
2. Enter schedule details:
   - **Name**: Descriptive name (e.g., "Morning Feed")
   - **Time**: When to dispense (24-hour format)
   - **Amount**: Feed amount in grams (1-1000g)
3. Use quick templates for common schedules
4. Test the schedule before saving

### Manual Feeding

1. Go to the **Dashboard**
2. Enter desired amount in the Manual Control section
3. Click **Dispense Feed**
4. Monitor the real-time status updates

### Monitoring and Logs

- **Dashboard**: View today's overview and recent activity
- **Logs**: Access complete history with filtering options
- **Real-time Updates**: Stats refresh automatically every 30 seconds

## Future Enhancements

The system is designed for easy integration of additional features:

### ML Feed Counter Integration
The modular architecture supports adding CSRNet or similar models for:
- Automatic feed level detection
- Dynamic amount adjustment based on consumption
- Predictive feeding recommendations

### Mobile App
- React Native or Flutter mobile application
- Push notifications
- Remote monitoring and control

### Advanced Analytics
- Feeding pattern analysis
- Health correlation tracking
- Consumption optimization

### Multi-Farm Support
- Farm management hierarchy
- Role-based access control
- Centralized monitoring dashboard

## Troubleshooting

### Common Issues

1. **Database errors**: Delete `chicken_feeder.db` to reset the database
2. **Port conflicts**: Change the port in `app.run(port=5001)`
3. **Permission errors**: Ensure write permissions for database file
4. **IoT connection issues**: Check device IP address and network connectivity

### Logs and Debugging

Enable debug mode for development:
```python
app.run(debug=True)
```

Check the console output for detailed error messages and stack traces.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues, feature requests, or questions:
- Create an issue in the repository
- Check existing documentation
- Review the troubleshooting section

---

**Happy Farming! 🐔🌾**
