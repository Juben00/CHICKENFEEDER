from picamera import PiCamera
from time import sleep
import os
import datetime

camera = PiCamera()

def capture_image():
    folder = "captures"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{folder}/feed_{timestamp}.jpg"
    print("Capturing image...")
    camera.start_preview()
    sleep(2)  # allow camera to adjust
    camera.capture(path)
    camera.stop_preview()
    print(f"Image saved at {path}")
    return path
