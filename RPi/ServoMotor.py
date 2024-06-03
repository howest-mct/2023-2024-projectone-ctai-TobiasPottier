import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)

class ServoMotor:
    def __init__(self) -> None:
        self.servopin = 18    
        GPIO.setup(self.servopin, GPIO.OUT)
        self.servo_pwm = GPIO.PWM(self.servopin, 50) # create PWM and set Frequency to 50Hz
        self.servo_pwm.start(3)

    def turn0degrees(self):
        self.servo_pwm.ChangeDutyCycle(3)
    def turn90degrees(self):
        self.servo_pwm.ChangeDutyCycle(8)
    def turn180degrees(self):
        self.servo_pwm.ChangeDutyCycle(13)
