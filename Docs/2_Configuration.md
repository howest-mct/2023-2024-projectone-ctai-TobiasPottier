# Project One: RPi

## Configuration

### Already done: Basic RPi setup:

1. **Download and write latest Raspberry Pi OS:**

   - Use Raspberry PI Imager to download the latest image
   - Set hostname, WiFi (Howest) and locale

### ⚠️ DO IT YOURSELF: Provide WiFi access on the Pi for home use:

1. **Gain Administrator Rights:**

   - Run `sudo -i` to gain administrator rights.

2. **Configure WiFi Access:**

   - Run `wpa_passphrase <your_SSID@Home> <your_wifi-password> >> /etc/wpa_supplicant/wpa_supplicant.conf`.
   - Replace `<your_SSID@Home>` with the name of your home network and `<your_wifi-password>` with the corresponding password.

3. **Reload Wireless Network Card:**

   - Run `wpa_cli -i wlan0 reconfigure` to reload your Pi's wireless network card.

     Or reboot...

4. **Test WiFi Connection:**
   - Run `wget www.google.com` to check if the wireless internet is working.

### Already Done: Full update/upgrade on May 19, 2024. If you want to upgrade, follow the steps below:

1. **Check for Available Updates:**

   - Run `apt update` to check for available updates.
   - (You still have sudo rights from the previous step, so no need to add `sudo`.)

2. **Install Available Updates:**

   - Run `apt upgrade` to install the available updates.

3. **Confirm Upgrade:**

   - Confirm with `Y` if prompted.

4. **Wait for Completion:**
   - Wait until the update is completed.

### ⚠️⚠️ Already Done: Install rpi-lgpio

Some of you may have noticed that after an update or a recent clean image (kernel >= 6.6), certain GPIO functions (like interrupts) no longer work with RPi.GPIO.

This can be resolved by using another library that follows the same syntax as the old RPi.GPIO library: rpi-lgpio.

We must not have both libraries installed, as they both create the RPi package.

1. **Remove Default RPi.GPIO Library:**

   - Run `apt remove python3-rpi.gpio -y` to remove the default RPi.GPIO library.

2. **Install New rpi-lgpio Library:**

   - Run `apt install python3-rpi-lgpio -y` to install the new rpi-lgpio library.

3. **Ensure Old RPi.GPIO Library is Not Installed via pip:**

   - Sometimes, the old RPi.GPIO library might get installed automatically with other libraries (like the Adafruit Neopixel library).

4. **Handle RuntimeError: Failed to add edge detection:**
   - If you encounter this error after installing another library, run the following commands within your virtual environment:
     1. Run `pip3 uninstall rpi-lgpio RPi.GPIO` to ensure everything is removed first.
     2. Run `pip3 install rpi-lgpio` to correctly install the new library.

### ⚠️ Already Done: Install BlueZ

Updated to the latest version of BlueZ, as seen in class or via https://github.com/bluez/bluez/blob/master/README

### ⚠️ Already Done: Install VSCode Python extension

Updated to the latest version of BlueZ, as seen in class or via https://github.com/bluez/bluez/blob/master/README

## TODO: see 1_Kickoff.md
