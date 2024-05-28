import threading
import queue
import LCD

from RPi import GPIO
import time
from ServoMotor import ServoMotor

GPIO.setmode(GPIO.BCM)
buzzer_pin = 4
GPIO.setup(buzzer_pin, GPIO.OUT)
buzzer_pwm = GPIO.PWM(buzzer_pin, 200)
buzzer_pwm.start(50)
buzzer_pwm.ChangeFrequency(1)

btn_pin = 16
GPIO.setup(btn_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
def door_btn(channel):
    global door_locked
    if GPIO.input(channel) == 0:
        print('Btn')
        door_locked = True
        
GPIO.add_event_detect(btn_pin, GPIO.BOTH, callback=door_btn, bouncetime=150)


lcd = LCD.LCD()
lcd.clear()

# Bluez gatt uart service (SERVER)
from bluetooth_uart_server.bluetooth_uart_server import ble_gatt_uart_loop

# extend this code so the value received via Bluetooth gets printed on the LCD
# (maybe together with you Bluetooth device name or Bluetooth MAC?)

def stop_buzzer():
    global buzzer_pwm, buzzer_pin
    buzzer_pwm.stop()
    GPIO.output(buzzer_pin, GPIO.LOW)

def start_buzzer(frequency):
    global buzzer_pwm, buzzer_pin
    buzzer_pwm = GPIO.PWM(buzzer_pin, frequency)
    buzzer_pwm.start(50)


MAC_ADDRESS = "D8:3A:DD:D9:73:57"
door_locked = True
connection_made = False
def main():
    global door_locked, connection_made
    i = 0
    rx_q = queue.Queue()
    tx_q = queue.Queue()
    device_name = "TPBias-pi-gatt-uart" # TODO: replace with your own (unique) device name
    threading.Thread(target=ble_gatt_uart_loop, args=(rx_q, tx_q, device_name), daemon=True).start()
    lcd.send_string("BLE Server Ready", lcd.LCD_LINE_1)

    servorMotor = ServoMotor()
    start_time = time.time()
    try:
        message = '__init__'
        while True:
            current_time = time.time()
            try:
                incoming = rx_q.get(timeout=.1) # Wait for up to .1 seconds
                if incoming:
                    message = "{}".format(incoming)
                    print(message)
                if message == 'Start':
                    connection_made = True
                    stop_buzzer()
                    lcd.clear()
                    lcd.send_string('Connected!', lcd.LCD_LINE_1)
                elif message == 'UD':
                    start_time = current_time
                    buzzer_pwm.ChangeFrequency(4)
                elif message == 'NUD':
                    lcd.clear()
                

            except Exception as e:
                pass # nothing in Q 
            if message == 'UD' and door_locked:
                lcd.backlight_on()
                timer = max((2 - abs(start_time - current_time)), 0)
                lcd.send_string('User Found!', lcd.LCD_LINE_1)
                if timer == 0:
                    door_locked = False
                else:
                    lcd.send_string(f'Auth... {timer:.2f}', lcd.LCD_LINE_2)
                    
            elif connection_made and door_locked:
                lcd.backlight_off()
                lcd.send_string(f'{" "*16}', lcd.LCD_LINE_2)
                servorMotor.turn0degrees()
                
            elif not door_locked:
                lcd.send_string(f'Door Unlocked!', lcd.LCD_LINE_2)
                servorMotor.turn180degrees()
            # if i%5 == 0: # Send some data every 5 iterations
            #     tx_q.put("test{}".format(i))
            # i += 1
    except Exception as ex:
        print(ex)
    finally:
        servorMotor.turn0degrees()
        time.sleep(.1)
        lcd.clear()
        time.sleep(.1)
        lcd.backlight_off()
        time.sleep(.1)
if __name__ == '__main__':
    main()
