/* 
ADC is set to ~32 kHz analog input using low level code found 
at https://forum.arduino.cc/t/faster-analog-read/6604/6
*/
int counter; 

#define FASTADC 1

// defines for setting and clearing register bits
#ifndef cbi
#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))
#endif
#ifndef sbi
#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit))
#endif

void setup() {
  int start ;
  int i ;
  
#if FASTADC
  // set prescale to 64 
  sbi(ADCSRA,ADPS2) ; // 1
  sbi(ADCSRA,ADPS1) ; // 1 
  cbi(ADCSRA,ADPS0) ; // 0 
#endif


// will print the analog input at the top of serial monitor
  Serial.begin(38400) ;
  delay(1000);
/*
  Serial.print("ADCTEST: ") ;
  start = millis() ;
  for (i = 0 ; i < 1000 ; i++)
    analogRead(0) ;
  Serial.print(1 / ((millis() - start) * 0.001)) ;
  Serial.println(" Hz analog input") ;
 */
}

void loop() 
{
  Serial.println(analogRead(0) - 500);
}
