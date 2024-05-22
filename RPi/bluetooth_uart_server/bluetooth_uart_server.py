import threading
import dbus, dbus.mainloop.glib
from gi.repository import GLib
from .utils_advertisement import Advertisement, register_ad_cb, register_ad_error_cb
from .utils_gatt_server import Service, Characteristic, register_app_cb, register_app_error_cb

BLUEZ_SERVICE_NAME =           'org.bluez'
DBUS_OM_IFACE =                'org.freedesktop.DBus.ObjectManager'
LE_ADVERTISING_MANAGER_IFACE = 'org.bluez.LEAdvertisingManager1'
GATT_MANAGER_IFACE =           'org.bluez.GattManager1'
GATT_CHRC_IFACE =              'org.bluez.GattCharacteristic1'
UART_SERVICE_UUID =            '6e400001-b5a3-f393-e0a9-e50e24dcca9e'
UART_RX_CHARACTERISTIC_UUID =  '6e400002-b5a3-f393-e0a9-e50e24dcca9e'
UART_TX_CHARACTERISTIC_UUID =  '6e400003-b5a3-f393-e0a9-e50e24dcca9e'
mainloop = None

class TxCharacteristic(Characteristic):
    def __init__(self, bus, index, service, tx_q, raw):
        Characteristic.__init__(self, bus, index, UART_TX_CHARACTERISTIC_UUID,
                                ['notify'], service)
        self.notifying = False
        self.tx_q = tx_q
        self.raw = raw
        if self.tx_q:
            threading.Thread(target=self.watch_tx_q, daemon=True).start()
        
    def watch_tx_q(self):
        while True:
            try:
                out = self.tx_q.get(timeout=1)
                if out:
                    # print("sending: {}".format(out))
                    self.send_tx(out)
            except Exception as e:
                pass # print("nothing to send")

    def send_tx(self, s):
        if not self.notifying:
            return
        value = []
        if not self.raw:
            for c in s:
                value.append(dbus.Byte(c.encode()))
        else:
            value = s
        self.PropertiesChanged(GATT_CHRC_IFACE, {'Value': value}, [])

    def StartNotify(self):
        if self.notifying:
            return
        self.notifying = True

    def StopNotify(self):
        if not self.notifying:
            return
        self.notifying = False

class RxCharacteristic(Characteristic):
    def __init__(self, bus, index, service, rx_q, raw):
        Characteristic.__init__(self, bus, index, UART_RX_CHARACTERISTIC_UUID,
                                ['write'], service)
        self.rx_q = rx_q
        self.raw = raw

    def WriteValue(self, value, options):
        # print('remote: {}'.format(bytearray(value).decode()))
        if self.rx_q:
            if not self.raw:
                self.rx_q.put(bytearray(value).decode())
            else:
                out = []
                for byte in bytearray(value):
                    # print(byte, hex(byte))
                    out.append(hex(byte))
                # print(out)
                # print(bytearray(value).decode())
                self.rx_q.put(out)

class UartService(Service):
    def __init__(self, bus, index, rx_q, tx_q, raw=False):
        Service.__init__(self, bus, index, UART_SERVICE_UUID, True)
        self.add_characteristic(TxCharacteristic(bus, 0, self, tx_q, raw))
        self.add_characteristic(RxCharacteristic(bus, 1, self, rx_q, raw))

class Application(dbus.service.Object):
    def __init__(self, bus):
        self.path = '/'
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}
        for service in self.services:
            response[service.get_path()] = service.get_properties()
            chrcs = service.get_characteristics()
            for chrc in chrcs:
                response[chrc.get_path()] = chrc.get_properties()
        return response


class UartApplication(Application):
    def __init__(self, bus, rx_q, tx_q, raw=False):
        Application.__init__(self, bus)
        self.add_service(UartService(bus, 0, rx_q, tx_q, raw))

class UartAdvertisement(Advertisement):
    def __init__(self, bus, index, device_name):
        Advertisement.__init__(self, bus, index, 'peripheral')
        self.add_service_uuid(UART_SERVICE_UUID)
        self.add_local_name(device_name)
        self.include_tx_power = True

def find_adapter(bus):
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'),
                               DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()
    for o, props in objects.items():
        if LE_ADVERTISING_MANAGER_IFACE in props and GATT_MANAGER_IFACE in props:
            return o
        print('Skip adapter:', o)
    return None

def ble_gatt_uart_loop(rx_q, tx_q, device_name='rpi-gatt-server', raw=False):
    global mainloop
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    adapter = find_adapter(bus)
    if not adapter:
        print('BLE adapter not found')
        return
    adapter_interface = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter), 'org.freedesktop.DBus.Properties')

    adapter_props = dbus.Interface(adapter_interface, 'org.freedesktop.DBus.Properties')
    mac_address = adapter_props.Get('org.bluez.Adapter1', 'Address')
    print(mac_address)


    service_manager = dbus.Interface(
                                bus.get_object(BLUEZ_SERVICE_NAME, adapter),
                                GATT_MANAGER_IFACE)
    ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter),
                                LE_ADVERTISING_MANAGER_IFACE)

    app = UartApplication(bus, rx_q, tx_q, raw)
    adv = UartAdvertisement(bus, 0, device_name)

    mainloop = GLib.MainLoop()

    service_manager.RegisterApplication(app.get_path(), {},
                                        reply_handler=register_app_cb,
                                        error_handler=register_app_error_cb)
    ad_manager.RegisterAdvertisement(adv.get_path(), {},
                                     reply_handler=register_ad_cb,
                                     error_handler=register_ad_error_cb)
    try:
        mainloop.run()
    except KeyboardInterrupt:
        adv.Release()