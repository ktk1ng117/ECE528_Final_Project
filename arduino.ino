const int pirInputPin = 7;
const int ledOutputPin = 3;
const int buzzerOutputPin = 5;
const int potPin = A6;
const int wakeUpOutputPin = 12;
const unsigned long cooldownTime = 15000; // 15 seconds cooldown
unsigned long lastSignalTime = 0; // Tracks the last time a signal was sent
int motion = 0;
int potVal = 0;
unsigned long motionStartTime = 0;
bool motionDetected = false;

void setup() {
  pinMode(ledOutputPin, OUTPUT);
  pinMode(pirInputPin, INPUT);
  pinMode(buzzerOutputPin, OUTPUT);
  pinMode(wakeUpOutputPin, OUTPUT);
  digitalWrite(wakeUpOutputPin, LOW);
  Serial.begin(9600);
}

void loop() {
  potVal = analogRead(potPin);
  motion = digitalRead(pirInputPin);

  if (motion) {
    if (!motionDetected) {
      motionStartTime = millis();
      motionDetected = true;
    }

    // Check if motion has been detected for at least 1.5 seconds
    if (motionDetected && millis() - motionStartTime >= 1500) {
      unsigned long currentTime = millis();
      if (currentTime - lastSignalTime >= cooldownTime) {
        digitalWrite(ledOutputPin, HIGH);
        digitalWrite(wakeUpOutputPin, HIGH);
        tone(buzzerOutputPin, potVal);
        delay(1000); // Buzzer beeps for 1 second
        noTone(buzzerOutputPin);
        digitalWrite(wakeUpOutputPin, LOW);
        lastSignalTime = currentTime; // Update the last signal time
      }
      motionDetected = false; // Reset motion detection
    }
  } else {
    motionDetected = false; // Reset if no motion
    digitalWrite(ledOutputPin, LOW);
  }

  // Check for data from Raspberry Pi
  if (Serial.available() > 0) {
    String data = Serial.readString();
    data.trim();
    readOut(data);
    if (data != "Non-Human") {
      tone(buzzerOutputPin, potVal);
      delay(5000); // Buzzer sounds for 5 seconds
      noTone(buzzerOutputPin);
    }
  }
}

void readOut(String dataOut) {
  Serial.println(dataOut);
}
