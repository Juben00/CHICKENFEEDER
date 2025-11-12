from gpiozero import Servo
from time import sleep

servo = Servo(17)  # GPIO17 (Pin 11)

def activate_servo():
    print("Activating servo motor...")
    servo.max()
    sleep(0.5)
    servo.min()
    sleep(0.5)
    servo.detach()
    print("Servo motor done dispensing.")
