#define led 13

void setup() {
  Serial.begin(9600);
  pinMode(led,OUTPUT);

}

void loop() {
  if(Serial.available() > 0){
    char comando = Serial.read();
    if(comando == '1'){
      digitalWrite(led, HIGH);
    } else if (comando == '0'){
      digitalWrite(led, LOW);
    }
  }
}
