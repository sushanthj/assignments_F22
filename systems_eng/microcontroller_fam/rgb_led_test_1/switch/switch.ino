// phasebutton is the one which iterates amongst defined 3 states
const int elementbuttonPin = 3;     // the number of the pushbutton pin
const int phasebuttonPin = 2;     // the number of the pushbutton pin
const int red_light_pin= 9;
const int green_light_pin = 10;
const int blue_light_pin = 11;
const int sensorPin = A0;
const int intermediary_pin = 4;

// Variables will change:
int redState = LOW;         // the current state of the output pin
int blueState = LOW;
int greenState = LOW;
int color_cycle[] = {LOW, HIGH, HIGH};
int phasebuttonState;             // the current reading from the input pin
int phaselastButtonState = LOW;   // the previous reading from the input pin

int elementbuttonState;             // the current reading from the input pin
int elementlastButtonState = HIGH;   // the previous reading from the input pin

int phase_track = 0;
int sensorValue = 0;
int intensity = 0;

char color;
volatile int state = LOW;

// the following variables are unsigned longs because the time, measured in
// milliseconds, will quickly become a bigger number than can be stored in an int.
unsigned long lastDebounceTime = 0;  // the last time the output pin was toggled
unsigned long debounceDelay = 50;    // the debounce time; increase if the output flickers

unsigned long elementlastDebounceTime = 0;  // the last time the output pin was toggled
unsigned long elementdebounceDelay = 50;    // the debounce time; increase if the output flickers

template <typename T>
Print& operator<<(Print& printer, T value)
{
    printer.print(value);
    return printer;
}

void setup() {
  pinMode(red_light_pin, OUTPUT);
  pinMode(green_light_pin, OUTPUT);
  pinMode(blue_light_pin, OUTPUT);
  // initialize the pushbutton pin as an input:
  // pinMode(phasebuttonPin, INPUT);
  pinMode(phasebuttonPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(phasebuttonPin), change_state, CHANGE);
  pinMode(elementbuttonPin, INPUT);
  Serial.begin(9600);
}

void change_state(){
  state = !state;
}

void loop() {
  // read the state of the switch into a local variable:
  // int reading = digitalRead(intermediary_pin);
  int reading = state;
  int element_reading = digitalRead(elementbuttonPin);
  sensorValue = analogRead(sensorPin);

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
        // bring LED to off state as default
        analog_color(255, 255, 255);
        // When phase change is detected do:
        change_phase();     
      }
    }
  }
  
  // If the switch changed, due to noise or pressing:
  if (element_reading != elementlastButtonState) {
    // reset the debouncing timer
    elementlastDebounceTime = millis();
  }

  if ((millis() - elementlastDebounceTime) > elementdebounceDelay) {
    // whatever the reading is at, it's been there for longer than the debounce
    // delay, so take it as the actual current state:

    // if the button state has changed:
    if (element_reading != elementbuttonState) {
      elementbuttonState = element_reading;

      // only toggle the LED if the new button state is HIGH
      if (elementbuttonState == HIGH) {
        // When phase change is detected do:
        statewise_actions();
      }
    }
  }

  // save the reading. Next time through the loop, it'll be the lastButtonState:
  phaselastButtonState = reading;

  // save the reading. Next time through the loop, it'll be the lastButtonState:
  elementlastButtonState = element_reading;

  if (phase_track == 1) {
    Serial.println(sensorValue);
    sensorValue = map(sensorValue, 0, 1023, 0, 255);
    analog_color(sensorValue, sensorValue, sensorValue); 
  }
  
  else if (phase_track  == 2) {
    if (Serial.available() > 0) {
    // read incoming serial data:
    String inpdata = Serial.readStringUntil('\n');
    inpdata.trim();
    intensity = inpdata.substring(1).toInt();
    color = inpdata.charAt(0);
    if (intensity < 256 && intensity > -1 && color == 'r') {
      Serial << "given color is " << color << " and given intensity is " << intensity;
      Serial.println(" ");
      analogWrite(red_light_pin, intensity);
      }
    else if (intensity < 256 && intensity > -1 && color == 'g') {
      Serial << "given color is " << color << " and given intensity is " << intensity;
      Serial.println(" ");
      analogWrite(green_light_pin, intensity);
      }
    else if (intensity < 256 && intensity > -1 && color == 'b') {
      Serial << "given color is " << color << " and given intensity is " << intensity;
      Serial.println(" ");
      analogWrite(blue_light_pin, intensity);
      }
    else {
        Serial.println("Wrong value given");
      }
    }
  }

}

void write_digital_state(int red_light_value, int green_light_value, int blue_light_value)
 {
  digitalWrite(red_light_pin, red_light_value);
  digitalWrite(green_light_pin, green_light_value);
  digitalWrite(blue_light_pin, blue_light_value);
}

void change_phase() {
  // execute the different States
  switch (phase_track) {
      case 0:
        Serial.println("Entered State 2");
        // write_digital_state(LOW, HIGH, LOW);   
        break;
      case 1:
        Serial.println("Entered State 3");
        // write_digital_state(HIGH, LOW, LOW);
        break;
      case 2:
        Serial.println("Entered State 1");
        // write_digital_state(LOW, LOW, HIGH);
        break;
      default:
        Serial.println("ERROR");
        break;
    }
  
  // update the states
  if (phase_track > 1) {
    phase_track = 0;
  }
  else {
    phase_track += 1;
  }
}

void statewise_actions() {
  switch (phase_track) {
      case 0:
        Serial.println("Currently in State 1");
        write_digital_state(color_cycle[0], color_cycle[1], color_cycle[2]);
        cycle_colors();
        break;
      case 1: 
        Serial.println("Currently in State 2");  
        break;
      case 2:
        Serial.println("Currently in State 3");
        // write_digital_state(LOW, LOW, HIGH);
        break;
      default:
        Serial.println("ERROR");
        break;
    }
}

void cycle_colors() {
  if (color_cycle[0] == LOW && color_cycle[1] == HIGH && color_cycle[2] == HIGH) {
    color_cycle[0] = HIGH;
    color_cycle[1] = LOW;
    color_cycle[2] = HIGH;
  }
  else if (color_cycle[0] == HIGH && color_cycle[1] == LOW && color_cycle[2] == HIGH) {
    color_cycle[0] = HIGH;
    color_cycle[1] = HIGH;
    color_cycle[2] = LOW;
  }
  else {
    color_cycle[0] = LOW;
    color_cycle[1] = HIGH;
    color_cycle[2] = HIGH;
  }
  
}

void analog_color(int red_light_value, int green_light_value, int blue_light_value)
 {
  analogWrite(red_light_pin, red_light_value);
  analogWrite(green_light_pin, green_light_value);
  analogWrite(blue_light_pin, blue_light_value);
}