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

lcd = LCD.LCD()
lcd.clear()

# Bluez gatt uart service (SERVER)
from bluetooth_uart_server.bluetooth_uart_server import ble_gatt_uart_loop

# extend this code so the value received via Bluetooth gets printed on the LCD
# (maybe together with you Bluetooth device name or Bluetooth MAC?)

MAC_ADDRESS = "D8:3A:DD:D9:73:57"

def main():
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
                    buzzer_pwm.ChangeFrequency(4)
                    lcd.clear()
                    lcd.send_string('Connected!', lcd.LCD_LINE_1)
                elif message == 'UD':
                    start_time = current_time
                    buzzer_pwm.ChangeFrequency(10)
                elif message == 'NUD':
                    lcd.clear()
                    buzzer_pwm.ChangeFrequency(4)
                

            except Exception as e:
                pass # nothing in Q 
            if message == 'UD':
                timer = max((2 - abs(start_time - current_time)), 0)
                lcd.send_string('User Found!', lcd.LCD_LINE_1)
                if timer == 0:
                    lcd.send_string(f'Door Unlocked!', lcd.LCD_LINE_2)
                    servorMotor.turn180degrees()
                else:
                    lcd.send_string(f'Auth... {timer:.2f}', lcd.LCD_LINE_2)
                    
            else:
                lcd.send_string(f'{" "*16}', lcd.LCD_LINE_2)
                servorMotor.turn0degrees()

            # if i%5 == 0: # Send some data every 5 iterations
            #     tx_q.put("test{}".format(i))
            # i += 1
    except Exception as ex:
        print(ex)
        pass
    finally:
        servorMotor.turn0degrees()
        time.sleep(.1)
        lcd.clear()
        time.sleep(.1)
if __name__ == '__main__':
    main()
