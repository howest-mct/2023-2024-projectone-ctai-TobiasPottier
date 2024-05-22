import threading
import queue
import LCD

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
    lcd.send_string(MAC_ADDRESS, lcd.LCD_LINE_2)
    try:
        while True:
            try:
                incoming = rx_q.get(timeout=1) # Wait for up to 1 second 
                if incoming:
                    message = "In main loop: {}".format(incoming)
                    print(message)
                    lcd.send_string(message, lcd.LCD_LINE_1)
            except Exception as e:
                pass # nothing in Q 

            # if i%5 == 0: # Send some data every 5 iterations
            #     tx_q.put("test{}".format(i))
            # i += 1
    except:
        pass
    finally:
        lcd.clear()
if __name__ == '__main__':
    main()


