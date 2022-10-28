// phasebutton is the one which iterates amongst defined 3 states
const int phasebuttonPin = 3;     // the number of the pushbutton pin
const int red_light_pin= 9;
const int green_light_pin = 10;
const int blue_light_pin = 11;

// Variables will change:
int redState = HIGH;         // the current state of the output pin
int blueState = HIGH;
int greenState = HIGH;
int phasebuttonState;             // the current reading from the input pin
int phaselastButtonState = LOW;   // the previous reading from the input pin

// the following variables are unsigned longs because the time, measured in
// milliseconds, will quickly become a bigger number than can be stored in an int.
unsigned long lastDebounceTime = 0;  // the last time the output pin was toggled
unsigned long debounceDelay = 50;    // the debounce time; increase if the output flickers

void setup() {
  pinMode(red_light_pin, OUTPUT);
  pinMode(green_light_pin, OUTPUT);
  pinMode(blue_light_pin, OUTPUT);
  // initialize the pushbutton pin as an input:
  pinMode(phasebuttonPin, INPUT);
  Serial.begin(9600);
  // set initial LED state
  // CHANGE THIS
  digitalWrite(red_light_pin, redState);
  digitalWrite(blue_light_pin, blueState);
  digitalWrite(green_light_pin, greenState);
}

void loop() {
  // read the state of the switch into a local variable:
  int reading = digitalRead(phasebuttonPin);

  // check to see if you just pressed the button
  // (i.e. the input went from LOW to HIGH), and you've waited long enough
  // since the last press to ignore any noise:

  // If the switch changed, due to noise or pressing:
  if (reading != phaselastButtonState) {
    // reset the debouncing timer
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    // whatever the reading is at, it's been there for longer than the debounce
    // delay, so take it as the actual current state:

    // if the button state has changed:
    if (reading != phasebuttonState) {
      phasebuttonState = reading;

      // only toggle the LED if the new button state is HIGH
      if (phasebuttonState == HIGH) {
        redState = !redState;
        greenState = !greenState;
        blueState = !blueState;
      }
    }
  }

  // set the LED:
  write_digital_state(redState, greenState, blueState);

  // save the reading. Next time through the loop, it'll be the lastButtonState:
  phaselastButtonState = reading;
  // switch (i) {
  //     case 0:
  //       Serial.println("Hello!");
  //       break;
  //     case 1:
  //       Serial.print("Let's learn ");
  //     case 2:
  //       Serial.println("Arduino");
  //       break;
  //     default:
  //       Serial.println("via ArduinoGetStarted.com");
  //       break;
  //   }
}

void write_digital_state(int red_light_value, int green_light_value, int blue_light_value)
 {
  digitalWrite(red_light_pin, red_light_value);
  digitalWrite(green_light_pin, green_light_value);
  digitalWrite(blue_light_pin, blue_light_value);
}
