from smbus import SMBus
import time

class LCD:
    I2C_ADDR = 0x27  # I2C device address
    LCD_CHR = 1       # Character mode
    LCD_CMD = 0       # Command mode
    LCD_LINE_1 = 0x80 # Instruction to go to beginning of line 1
    LCD_LINE_2 = 0xC0 # Instruction to go to beginning of line 2
    LCD_BACKLIGHT_ON = 0x08  # Data bit value to turn backlight on
    LCD_BACKLIGHT_OFF = 0x00 # Data bit value to turn backlight off
    ENABLE = 0b00000100    # Enable bit value
    E_DELAY = 0.0005       # Delay between pulses

    def __init__(self):
        self.bus = SMBus(1)
        self.backlight = self.LCD_BACKLIGHT_ON
        self.lcd_init()

    def lcd_init(self):
        self.send_byte(0x33, self.LCD_CMD) # initializing the LCD in 4-bit mode.
        self.send_byte(0x32, self.LCD_CMD) # initializing the LCD in 4-bit mode.
        self.send_byte(0x06, self.LCD_CMD) # entry mode, specifying how the cursor moves and whether the display shifts.
        self.send_byte(0x0C, self.LCD_CMD) # turns off the cursor.
        self.send_byte(0x28, self.LCD_CMD) # configures display lines and character font.
        self.send_byte(0x01, self.LCD_CMD) # Clears the display.
        time.sleep(0.05)

    def send_byte(self, byte, mode):
        bits_high = mode | (byte & 0xF0) | self.backlight
        bits_low = mode | ((byte & 0x0F) << 4) | self.backlight
        self.bus.write_byte(self.I2C_ADDR, bits_high)
        self.send_byte_with_e_toggle(bits_high)
        self.bus.write_byte(self.I2C_ADDR, bits_low)
        self.send_byte_with_e_toggle(bits_low)

    def send_byte_with_e_toggle(self, bits):
        time.sleep(self.E_DELAY)
        self.bus.write_byte(self.I2C_ADDR, (bits | self.ENABLE))
        time.sleep(self.E_DELAY)
        self.bus.write_byte(self.I2C_ADDR, (bits & ~self.ENABLE))
        time.sleep(self.E_DELAY)

    def send_instruction(self, instruction):
        self.send_byte(instruction, self.LCD_CMD)

    def send_character(self, char):
        self.send_byte(ord(char), self.LCD_CHR)

    def send_string(self, message, line=LCD_LINE_1):
        self.send_instruction(line)
        for char in message:
            self.send_character(char)

    def clear(self):
        self.send_instruction(0x01)

    def display_on(self):
        self.send_instruction(0x0C)

    def display_off(self):
        self.send_instruction(0x08)

    def cursor_on(self):
        self.send_instruction(0x0E)

    def cursor_off(self):
        self.send_instruction(0x0C)

    def backlight_on(self):
        self.backlight = self.LCD_BACKLIGHT_ON
        self.send_byte(0x00, self.LCD_CMD)  # Dummy write to apply backlight change

    def backlight_off(self):
        self.backlight = self.LCD_BACKLIGHT_OFF
        self.send_byte(0x00, self.LCD_CMD)  # Dummy write to apply backlight change


