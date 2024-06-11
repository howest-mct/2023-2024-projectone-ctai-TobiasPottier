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
tx_buzzer = queue.Queue()
def buzzer(rx_q):
    global buzzer_pwm
    previous_message = ''
    while True:
        if rx_q is not None:
            try:
                incoming = rx_q.get(timeout=.1) # Wait for up to .1 seconds
                if incoming:
                    message = "{}".format(incoming)
                if message != previous_message:
                    if message == 'play_buzzer':
                        start_buzzer(1)
                    elif message == 'stop_buzzer':
                        stop_buzzer()
                previous_message = message
            except Exception as ex:
                time.sleep(.1)

threading.Thread(target=buzzer, args=(tx_buzzer,)).start()

btn_pin = 16
GPIO.setup(btn_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
def door_btn(channel):
    global door_locked, start_time, current_time, open_door
    if GPIO.input(channel) == 0:
        if door_locked == False:
            door_locked = True
        else:
            open_door = True
        start_time = current_time
        
        
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

def start_buzzer(frequency):
    global buzzer_pwm, buzzer_pin
    buzzer_pwm.start(50)


MAC_ADDRESS = "D8:3A:DD:D9:73:57"
door_locked = True
connection_made = False
start_time = time.time()
current_time = time.time()
open_door = False

def main():
    global door_locked, connection_made, start_time, current_time, open_door
    i = 0
    rx_q = queue.Queue()
    tx_q = queue.Queue()
    device_name = "TPBias-pi-gatt-uart" # your own (unique) device name
    threading.Thread(target=ble_gatt_uart_loop, args=(rx_q, tx_q, device_name), daemon=True).start()

    servorMotor = ServoMotor()
    message = 'NUD'
    try:
        while True:
            current_time = time.time()
            try:
                incoming = rx_q.get(timeout=.1) # Wait for up to .1 seconds
                if incoming:
                    message = "{}".format(incoming)
                    print(message)
                if message == 'Start':
                    connection_made = True
                    tx_buzzer.put('stop_buzzer')
                    lcd.clear()
                    lcd.send_string('Connected!', lcd.LCD_LINE_1)
                elif message == 'UD':
                    if not open_door:
                        start_time = current_time
                elif message == 'NUD':
                    lcd.clear()
                elif message == 'exit':
                    connection_made = False
                    tx_buzzer.put('play_buzzer')
            except Exception as e:
                pass # nothing in Q 


            if door_locked and connection_made:
                if open_door:
                    lcd.send_string(f'{" "*16}', lcd.LCD_LINE_1)
                    lcd.send_string(f'{" "*16}', lcd.LCD_LINE_2)
                    lcd.backlight_on()
                    servorMotor.turn180degrees()
                    timer = max((10 - abs(start_time - current_time)), 0)
                    if timer == 0:
                        open_door = False
                        start_time = current_time
                    else:
                        lcd.send_string(f'Closing... {timer:.2f}', lcd.LCD_LINE_2)
                else:
                    servorMotor.turn0degrees()
                    lcd.send_string(f'{" "*16}', lcd.LCD_LINE_1)
                    lcd.send_string(f'{" "*16}', lcd.LCD_LINE_2)
                    if message == 'UD':
                        lcd.backlight_on()
                        timer = max((2 - abs(start_time - current_time)), 0)
                        lcd.send_string('User Found!', lcd.LCD_LINE_1)
                        if timer == 0:
                            door_locked = False
                        else:
                            lcd.send_string(f'Auth... {timer:.2f}', lcd.LCD_LINE_2)
                    else:
                        lcd.backlight_off()
                        lcd.send_string(f'{" "*16}', lcd.LCD_LINE_2)
            elif not door_locked and connection_made:
                lcd.send_string(f'{" "*16}', lcd.LCD_LINE_1)
                lcd.send_string(f'{" "*16}', lcd.LCD_LINE_2)
                lcd.send_string(f'Door Unlocked!', lcd.LCD_LINE_2)
                servorMotor.turn180degrees()
            elif not connection_made:
                servorMotor.turn0degrees()
                lcd.clear()
                lcd.send_string("No Connection", lcd.LCD_LINE_1)
            # if i%5 == 0: # Send some data every 5 iterations
            #     tx_q.put("test{}".format(i))
            # i += 1
    except Exception as ex:
        print(ex)
    finally:
        servorMotor.turn0degrees()
        time.sleep(.2)
        lcd.clear()
        time.sleep(.1)
        lcd.backlight_off()
        time.sleep(.1)
if __name__ == '__main__':
    main()
