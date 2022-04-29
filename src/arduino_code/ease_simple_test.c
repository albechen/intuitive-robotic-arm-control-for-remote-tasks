#include <Arduino.h>
#define USE_PCA9685_SERVO_EXPANDER
#include "ServoEasing.hpp"
#define MAX_EASING_SERVOS 16

#define START_DEGREE_VALUE 90
// #define END_DEGREE_VALUE 45
#define FIRST_PCA9685_EXPANDER_ADDRESS PCA9685_DEFAULT_ADDRESS
#define NUMBER_OF_SERVOS MAX_EASING_SERVOS

void getAndAttach16ServosToPCA9685Expander(uint8_t aPCA9685I2CAddress)
{
    ServoEasing *tServoEasingObjectPtr;

    Serial.print(F("Get ServoEasing objects and attach servos to PCA9685 expander at address=0x"));
    Serial.println(aPCA9685I2CAddress, HEX);
    for (uint_fast8_t i = 0; i < PCA9685_MAX_CHANNELS; ++i)
    {
#if defined(ARDUINO_SAM_DUE)
        tServoEasingObjectPtr = new ServoEasing(aPCA9685I2CAddress, &Wire1);
#else
        tServoEasingObjectPtr = new ServoEasing(aPCA9685I2CAddress, &Wire);
#endif
        if (tServoEasingObjectPtr->attach(i) == INVALID_SERVO)
        {
            Serial.print(F("Address=0x"));
            Serial.print(aPCA9685I2CAddress, HEX);
            Serial.print(F(" i="));
            Serial.print(i);
            Serial.println(
                F(
                    " Error attaching servo - maybe MAX_EASING_SERVOS=" STR(MAX_EASING_SERVOS) " is to small to hold all servos"));
        }
    }
}

void getAndAttach16ServosToPCA9685Expander(uint8_t aPCA9685I2CAddress);

void setup()
{
    Serial.begin(9600);
    checkI2CConnection(FIRST_PCA9685_EXPANDER_ADDRESS, &Serial);
    getAndAttach16ServosToPCA9685Expander(FIRST_PCA9685_EXPANDER_ADDRESS);
    setSpeedForAllServos(10);
    writeAllServos(START_DEGREE_VALUE);
    delay(500);
}

void loop()
{
    setSpeedForAllServos(30); // This speed is taken if no further speed argument is given.
    ServoEasing::ServoEasingArray[0]->startEaseTo(45);
    ServoEasing::ServoEasingArray[3]->startEaseTo(45);
    ServoEasing::ServoEasingArray[7]->startEaseTo(45);
    delay(3000);
    ServoEasing::ServoEasingArray[0]->startEaseTo(90);
    ServoEasing::ServoEasingArray[3]->startEaseTo(130);
    ServoEasing::ServoEasingArray[7]->startEaseTo(130);
    delay(3000);
    ServoEasing::ServoEasingArray[0]->startEaseTo(45);
    ServoEasing::ServoEasingArray[3]->startEaseTo(15);
    ServoEasing::ServoEasingArray[7]->startEaseTo(15);
    delay(3000);
    ServoEasing::ServoEasingArray[0]->startEaseTo(START_DEGREE_VALUE);
    ServoEasing::ServoEasingArray[3]->startEaseTo(START_DEGREE_VALUE);
    ServoEasing::ServoEasingArray[7]->startEaseTo(START_DEGREE_VALUE);
    delay(3000);
}