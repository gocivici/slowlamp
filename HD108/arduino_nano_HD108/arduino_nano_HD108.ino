#include <SPI.h>

// Clock and data pins are whatever are SPI defaults for your board (SCK, MOSI)
// Arduino Mega 2560, Clock 52, Data 51
// Nano: 
// SCK: D13: clock
// MOSI: D11: data
//common ground  

uint16_t num_leds = 55;

uint16_t getBrightness(uint8_t rb, uint8_t gb, uint8_t bb) {
  uint8_t r5 = rb >> 3;
  uint8_t g5 = gb >> 3;
  uint8_t b5 = bb >> 3;
  return (1 << 15) | (r5 << 10) | (g5 << 5) | b5;
}

void sendPixel(uint16_t brightness, uint16_t r, uint16_t g, uint16_t b) {
  // Brightness frame = 16 bits
  SPI.transfer16(brightness);

  // Color frame
  SPI.transfer16(r);
  SPI.transfer16(g);
  SPI.transfer16(b);
}

void setup() {
  SPI.begin();
}
void loop() {
  SPI.beginTransaction(SPISettings(500000, MSBFIRST, SPI_MODE3));
  // Start frame
  for (int i = 0; i <=  8; i++) {SPI.transfer16(0);} //4
  // LEDs
  // for (int i = 0; i <= 72 * 5; i++) {
  for (int i = 0; i < num_leds; i++){
    // LED frame
    // SPI.transfer(0xFF);  // Start of LED Frame & Brightness
    // SPI.transfer(0xFF);  // as (1)(5bit)(5bit)(5bit) brightnesses
    // Each frame = 2 bytes global brightness + 2*R + 2*G + 2*B = 8 bytes

    // SPI.transfer16(0xFFFF); //full brightness
    // SPI.transfer16(0b1011110111101111); // half brightness 
    // if (i%2 == 1){
    //   SPI.transfer16(0x00);  // RED (16-bit)
    //   SPI.transfer16(0xFFFF);  // GREEN (16-bit)
    //   SPI.transfer16(0xFFFF);  // BLUE (16-bit)
    // }
    // else{
    //   SPI.transfer16(0xFFFF); // RED (16-bit)
    //   SPI.transfer16(0x00);  // GREEN (16-bit)
    //   SPI.transfer16(0x00);  // BLUE (16-bit)
    // }
    if (i%2 == 1){
      uint16_t brightness = getBrightness(15, 15, 15);
      sendPixel(brightness, 0, 0, 0xFFFF ); //blue
    }else{
      uint16_t brightness = getBrightness(0, 0, 31);
      sendPixel(brightness, 0xFFFF/2, 0xFFFF/2, 0xFFFF/2 ); // a bit paler blue??
      // sendPixel(brightness, 0xFFFF/2, 0, 0xFFFF/2 );  
    } 
  }
  // End Frame
  for (int i = 0; i <= num_leds*2; i++) {SPI.transfer16(0xFFFF);} //? 
  SPI.endTransaction();

  delay(1000);
}