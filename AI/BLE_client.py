import asyncio
from itertools import count, takewhile
import sys
import time
from datetime import datetime
from typing import Iterator

from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

import queue

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

BLE_DEVICE_MAC = "D8:3A:DD:D9:73:57"  # Change to your device's MAC address

# TIP: you can get this function and more from the ``more-itertools`` package.
def sliced(data: bytes, n: int) -> Iterator[bytes]:
    """
    Slices *data* into chunks of size *n*. The last slice may be smaller than
    *n*.
    """
    return takewhile(len, (data[i: i + n] for i in count(0, n)))


async def uart_terminal(rx_q=None, tx_q=None, targetDeviceName=None, targetDeviceMac=None):
    """This is a simple "terminal" program that uses the Nordic Semiconductor
    (nRF) UART service. It reads from stdin and sends each line of data to the
    remote device. Any data received from the device is printed to stdout.
    """
    def match_nus_uuid(device: BLEDevice, adv: AdvertisementData):
        if targetDeviceMac != None and device.address == targetDeviceMac and UART_SERVICE_UUID.lower() in adv.service_uuids:
            return True
        if targetDeviceName != None and device.name == targetDeviceName and UART_SERVICE_UUID.lower() in adv.service_uuids:
            return True
        return False
    
    device = None
    try:
        while device is None:
            print('Searching Device...')
            device = await BleakScanner.find_device_by_filter(match_nus_uuid)
            if device is None:
                print("No matching device found, Trying again...")
    except:
        print('>Manual Exit<')
        sys.exit()    
        


    print("Found device, connecting...")

    def handle_disconnect(_: BleakClient):
        print("Device was disconnected, goodbye.")
        for task in asyncio.all_tasks():
            task.cancel()

    async with BleakClient(device, disconnected_callback=handle_disconnect) as client:
        try:
            print("Connected")
            await client.start_notify(UART_TX_CHAR_UUID, lambda _: None)  # No-op handler for notifications

            nus = client.services.get_service(UART_SERVICE_UUID)
            rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)

            personal_data = "Beep"

            if tx_q is not None:
                tx_q.put(personal_data)

            # Additional logic for continuous operation if needed
            while True:
                print(tx_q)
                if tx_q is not None:
                    try:
                        data = tx_q.get_nowait()
                        if data is not None:
                            data = data.encode()
                            for s in sliced(data, rx_char.max_write_without_response_size):
                                await client.write_gatt_char(rx_char, s, response=False)
                            print("Sent:", data.decode())
                    except:
                        time.sleep(0.5)
                        pass
                else:
                    await asyncio.sleep(.2)

        except KeyboardInterrupt:
            print("Disconnecting...")

def run(rx_q=None, tx_q=None, targetDeviceName=None, targetDeviceMac=None):
    if targetDeviceName is None and targetDeviceMac is None:
        raise ValueError("Both targetDeviceName and targetDeviceMac cannot be None. Please provide at least one.")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(uart_terminal(
            rx_q=rx_q, tx_q=tx_q, targetDeviceName=targetDeviceName, targetDeviceMac=targetDeviceMac))
    except asyncio.exceptions.CancelledError:
        pass


if __name__ == '__main__':
    # Create a queue for transmitting data
    tx_queue = queue.Queue()
    run(tx_q=tx_queue, targetDeviceMac=BLE_DEVICE_MAC)
