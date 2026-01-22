//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Serial Output
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! UART serial port driver for debug output.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use core::fmt::{self, Write};
use spin::Mutex;
use x86_64::instructions::port::Port;

/// COM1 port address
const COM1: u16 = 0x3F8;

/// Serial port wrapper
struct SerialPort {
    data: Port<u8>,
    line_status: Port<u8>,
}

impl SerialPort {
    const fn new(base: u16) -> Self {
        Self {
            data: Port::new(base),
            line_status: Port::new(base + 5),
        }
    }
    
    fn init(&mut self) {
        unsafe {
            // Disable interrupts
            Port::<u8>::new(COM1 + 1).write(0x00);
            // Enable DLAB
            Port::<u8>::new(COM1 + 3).write(0x80);
            // Set baud rate divisor to 1 (115200)
            Port::<u8>::new(COM1).write(0x01);
            Port::<u8>::new(COM1 + 1).write(0x00);
            // 8 bits, no parity, one stop bit
            Port::<u8>::new(COM1 + 3).write(0x03);
            // Enable FIFO
            Port::<u8>::new(COM1 + 2).write(0xC7);
            // Enable DTR/RTS
            Port::<u8>::new(COM1 + 4).write(0x0B);
        }
    }
    
    fn is_transmit_empty(&mut self) -> bool {
        unsafe { self.line_status.read() & 0x20 != 0 }
    }
    
    fn write_byte(&mut self, byte: u8) {
        while !self.is_transmit_empty() {}
        unsafe { self.data.write(byte); }
    }
}

impl Write for SerialPort {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            self.write_byte(byte);
        }
        Ok(())
    }
}

/// Global serial port
static SERIAL: Mutex<SerialPort> = Mutex::new(SerialPort::new(COM1));

/// Initialize serial port
pub fn init() {
    SERIAL.lock().init();
}

/// Print to serial
#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use core::fmt::Write;
    SERIAL.lock().write_fmt(args).unwrap();
}

/// Print macro
#[macro_export]
macro_rules! serial_print {
    ($($arg:tt)*) => ($crate::serial::_print(format_args!($($arg)*)));
}

/// Println macro
#[macro_export]
macro_rules! serial_println {
    () => ($crate::serial_print!("\n"));
    ($($arg:tt)*) => ($crate::serial_print!("{}\n", format_args!($($arg)*)));
}
