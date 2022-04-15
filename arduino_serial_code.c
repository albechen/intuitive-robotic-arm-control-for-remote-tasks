#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
// you can also call it with a different address you want
// Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x41);

// Depending on your servo make, the pulse width min and max may vary, you
// want these to be as small/large as possible without hitting the hard stop
// for max range. You'll have to tweak them as necessary to match the servos you
// have!
#define SERVOMIN 150 // this is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX 600 // this is the 'maximum' pulse length count (out of 4096)

int maxServoDegree = 60;
int maxOutput = (SERVOMAX - SERVOMIN) * maxServoDegree / 180 + SERVOMIN;

float tempFloat = 0;
int incoming[2];
int mapper[2];
int priorIncoming[2];

void setup()
{
    Serial.begin(9600);
    Serial.println("START TRACKING");

    pwm.begin();

    pwm.setPWMFreq(60); // Analog servos run at ~60 Hz updates

    yield();
}

void loop()
{
    while (Serial.available() >= 3)
    {
        for (int i = 0; i < 3; i++)
        {
            incoming[i] = Serial.read() * (maxOutput - SERVOMIN) / 100 + SERVOMIN;
        }
        pwm.setPWM(0, 0, incoming[0]);
        pwm.setPWM(2, 0, incoming[1]);
        pwm.setPWM(4, 0, incoming[2]);
        delay(200);
    }
}

// Serial.print(incoming[0]);
// Serial.println(incoming[1]);
// Serial.println(incoming[2]);
