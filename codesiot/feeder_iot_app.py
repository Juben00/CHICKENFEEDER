# Raspberry Pi IoT Feeder Flask App
# This app receives feed dispense commands and controls hardware (e.g., GPIO)

from flask import Flask, request, jsonify
import RPi.GPIO as GPIO
import time

app = Flask(__name__)

# GPIO setup (example: pin 18 for motor/servo)
FEEDER_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(FEEDER_PIN, GPIO.OUT)

@app.route('/dispense', methods=['POST'])
def dispense():
    data = request.get_json()
    amount_grams = data.get('amount', 0)
    # Implement hardware control logic here
    try:
        # Example: activate motor for a duration based on amount_grams
        duration = max(1, int(amount_grams / 10))  # Simple mapping
        GPIO.output(FEEDER_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(FEEDER_PIN, GPIO.LOW)
        return jsonify({'success': True, 'message': f'Dispensed {amount_grams}g'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    # Optionally return device status, uptime, etc.
    return jsonify({'status': 'online'})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001)
    finally:
        GPIO.cleanup()
