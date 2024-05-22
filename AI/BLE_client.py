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

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

BLE_DEVICE_MAC = "D8:3A:DD:B8:9E:86" #"C4:4F:33:75:D9:B3"  # "D8:3A:DD:B8:9E:86" for RPi


# TIP: you can get this function and more from the ``more-itertools`` package.
def sliced(data: bytes, n: int) -> Iterator[bytes]:
    """
    Slices *data* into chunks of size *n*. The last slice may be smaller than
    *n*.
    """
    return takewhile(len, (data[i: i + n] for i in count(0, n)))


async def uart_terminal(rx_q=None, tx_q=None, targetDeviceName=None, targetDeviceMac=None):
    # try:
    """This is a simple "terminal" program that uses the Nordic Semiconductor
    (nRF) UART service. It reads from stdin and sends each line of data to the
    remote device. Any data received from the device is printed to stdout.
    """
    def match_nus_uuid(device: BLEDevice, adv: AdvertisementData):
        # This assumes that the device includes the UART service UUID in the
        # advertising data. This test may need to be adjusted depending on the
        # actual advertising data supplied by the device.
        print("found", device.address)
        if targetDeviceMac != None and device.address == targetDeviceMac and UART_SERVICE_UUID.lower() in adv.service_uuids:   # ESP
            return True
        if targetDeviceName != None and device.name == targetDeviceName and UART_SERVICE_UUID.lower() in adv.service_uuids:   # ESP
            return True
        return False

    device = await BleakScanner.find_device_by_filter(match_nus_uuid)

    if device is None:
        print("no matching device found, you may need to edit targetDeviceMac or targetDeviceName or match_nus_uuid().")
        sys.exit(1)

    print("found device, connecting....")

    def handle_disconnect(_: BleakClient):
        print("Device was disconnected, goodbye.")
        # cancelling all tasks effectively ends the program
        for task in asyncio.all_tasks():
            task.cancel()

    def disconnect():
        print("Disconnecting....")

    def handle_rx(_: BleakGATTCharacteristic, data: bytearray):
        print("received:", data)
        if (rx_q != None):
            rx_q.put(data)

    async with BleakClient(device, disconnected_callback=handle_disconnect) as client:
        try:
            print("connected")
            await client.start_notify(UART_TX_CHAR_UUID, handle_rx)

            nus = client.services.get_service(UART_SERVICE_UUID)
            rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)
            if tx_q == None: # DEMO
                time.sleep(1)
                last_dt = ""
                # try:
                while True:
                    now = datetime.now()
                    timeString = now.strftime("%M%S")
                    if last_dt != timeString:
                        last_dt = timeString
                        data = timeString.encode()
                        for s in sliced(data, rx_char.max_write_without_response_size):
                            await client.write_gatt_char(rx_char, s, response=False)
                        print("sent:", data)
                    time.sleep(0.2)
            else:
                while True:
                    

                    try:
                        data = tx_q.get_nowait()
                        if data != None:
                            # print("got q data {}".format(data))
                            data = data.encode()
                            for s in sliced(data, rx_char.max_write_without_response_size):
                                await client.write_gatt_char(rx_char, s, response=False)
                            print("sent:", data)
                    except:  # no data
                        time.sleep(0.5)
                        pass

        except KeyboardInterrupt:
            disconnect()


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
    run(targetDeviceMac=BLE_DEVICE_MAC)
