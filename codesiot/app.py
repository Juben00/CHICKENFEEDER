from flask import Flask, jsonify, request
import requests, json, os
from servo import activate_servo
from camera import capture_image

app = Flask(__name__)

# Load configuration
with open("config.json") as f:
    config = json.load(f)

UPLOAD_ENDPOINT = config["upload_endpoint"]
DEVICE_ID = config["device_id"]
USER_TOKEN = config["user_token"]

@app.route('/')
def home():
    return jsonify({"message": f"IoT device {DEVICE_ID} online."})

@app.route('/activate_servo', methods=['POST'])
def servo_route():
    activate_servo()
    return jsonify({"status": "success", "message": "Servo activated."})

@app.route('/capture_image', methods=['POST'])
def capture_route():
    image_path = capture_image()
    return jsonify({"status": "success", "image_path": image_path})

@app.route('/upload_feed_image', methods=['POST'])
def upload_feed_image():
    """Capture image then upload to website API"""
    image_path = capture_image()

    with open(image_path, 'rb') as img:
        files = {'image': img}
        data = {'device_id': DEVICE_ID}
        headers = {'Authorization': f'Bearer {USER_TOKEN}'}

        try:
            res = requests.post(UPLOAD_ENDPOINT, files=files, data=data, headers=headers)
            if res.status_code == 200:
                return jsonify({"status": "uploaded", "response": res.json()})
            else:
                return jsonify({"status": "failed", "error": res.text}), 500
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
