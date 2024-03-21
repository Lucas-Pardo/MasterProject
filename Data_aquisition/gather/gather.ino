// int numSamples=0;
// long t, t0;
// uint16_t buf[256]; // 256*2 bytes buffer for voltage
// unsigned long buft[256]; // 256*4 bytes buffer for time
volatile int bufn;
int x;
// float vrange = 2.56;

void setup()
{
  // Serial.begin(2000000);
  Serial.begin(115200);

  ADCSRA = 0;             // clear ADCSRA register
  ADCSRB = 0;             // clear ADCSRB register
  ADMUX |= (0 & 0x07);    // set A0 analog input pin
  ADMUX |= (1 << REFS0);  // set reference voltage: Uses AREF pin (if only this bit is set)
  // ADMUX |= (1 << REFS1);  // set reference voltage: 2.56V (if both bits are set)
  ADMUX |= (1 << ADLAR);  // left align ADC value to 8 bits from ADCH register

  // sampling rate is [ADC clock] / [prescaler] / [conversion clock cycles]
  // for Arduino Uno ADC clock is 16 MHz and a conversion takes 13 clock cycles
  //ADCSRA |= (1 << ADPS2) | (1 << ADPS0);    // 32 prescaler for 38.5 KHz
  ADCSRA |= (1 << ADPS2);                     // 16 prescaler for 76.9 KHz
  //ADCSRA |= (1 << ADPS1) | (1 << ADPS0);    // 8 prescaler for 153.8 KHz

  ADCSRA |= (1 << ADATE); // enable auto trigger
  ADCSRA |= (1 << ADIE);  // enable interrupts when measurement complete
  ADCSRA |= (1 << ADEN);  // enable ADC
  ADCSRA |= (1 << ADSC);  // start ADC measurements

  bufn = 0;
}

ISR(ADC_vect)
{
  // buft[bufn] = micros();
  // buf[bufn++] = ADCH * 256 + ADCL;
  // x = (ADCH * 256 + ADCL) * vrange / 1024; // True voltage
  x = ADCH;  // read 8 bit value from ADC
  // numSamples++;
}
  
void loop()
{
  Serial.print("Reading:");
  // Serial.println(x - vrange/2);
  Serial.println(x);

  // if (bufn == 256){ // Buffer full
  //   Serial.write((uint8_t *) buft, 1024); // Send times: 256*4 = 1024 bytes
  //   Serial.write((uint8_t *) buf, 512); // Send voltages: 256*2 = 512 bytes
  //   bufn = 0; // Reset pointer to overwrite buffers
  // }


  // if (numSamples>=1000)
  // {
  //   t = micros()-t0;  // calculate elapsed time

  //   Serial.print("Sampling frequency: ");
  //   Serial.print((float)1000000/t);
  //   Serial.println(" KHz");
  //   delay(2000);
    
  //   // restart
  //   t0 = micros();
  //   numSamples=0;
  // }
}