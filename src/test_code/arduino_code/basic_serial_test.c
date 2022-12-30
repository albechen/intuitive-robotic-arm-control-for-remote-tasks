#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
#define SERVOMIN 150 // this is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX 620 // this is the 'maximum' pulse length count (out of 4096)

int incoming[2];

void setup()
{
    Serial.begin(9600);
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
            incoming[i] = map(Serial.read(), 0, 180, SERVOMIN, SERVOMAX);
        }
        pwm.setPWM(0, 0, incoming[0]);
        pwm.setPWM(3, 0, incoming[1]);
        pwm.setPWM(7, 0, incoming[2]);
    }
}

// Serial.print(incoming[0]);
// Serial.println(incoming[1]);
// Serial.println(incoming[2]);