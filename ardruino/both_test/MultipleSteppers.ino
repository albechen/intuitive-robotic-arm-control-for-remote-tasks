// MultiStepper.pde
// -*- mode: C++ -*-
//
// Shows how to multiple simultaneous steppers
// Runs one stepper forwards and backwards, accelerating and decelerating
// at the limits. Runs other steppers at the same time
//
// Copyright (C) 2009 Mike McCauley
// $Id: MultiStepper.pde,v 1.1 2011/01/05 01:51:01 mikem Exp mikem $

#include <AccelStepper.h>
 #include <Arduino.h>
#define USE_PCA9685_SERVO_EXPANDER
#include "ServoEasing.hpp"

#define START_DEGREE_VALUE 0
#define START_DEGREE 0
#define END_DEGREE 90
#define SERVO_SPEED 100 
// #define END_DEGREE_VALUE 45
#define FIRST_PCA9685_EXPANDER_ADDRESS PCA9685_DEFAULT_ADDRESS
#define NUMBER_OF_SERVOS MAX_EASING_SERVOS

void getAndAttach16ServosToPCA9685Expander(uint8_t aPCA9685I2CAddress) {
    ServoEasing *tServoEasingObjectPtr;

    Serial.print(F("Get ServoEasing objects and attach servos to PCA9685 expander at address=0x"));
    Serial.println(aPCA9685I2CAddress, HEX);
    for (uint_fast8_t i = 0; i < PCA9685_MAX_CHANNELS; ++i) {
#if defined(ARDUINO_SAM_DUE)
        tServoEasingObjectPtr= new ServoEasing(aPCA9685I2CAddress, &Wire1);
#else
        tServoEasingObjectPtr = new ServoEasing(aPCA9685I2CAddress, &Wire);
#endif
        if (tServoEasingObjectPtr->attach(i) == INVALID_SERVO) {
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

#define MAX_SERVOS_USED 7
int incoming[MAX_SERVOS_USED];


// Define some steppers and the pins the will use
AccelStepper stepper1(AccelStepper::FULL4WIRE, 23, 27, 25, 29);
//AccelStepper stepper2(AccelStepper::FULL4WIRE, 33, 37, 35, 39)
AccelStepper stepper3(AccelStepper::FULL4WIRE, 43, 47, 45, 49);

void setup() {
    checkI2CConnection(FIRST_PCA9685_EXPANDER_ADDRESS, &Serial);
    getAndAttach16ServosToPCA9685Expander(FIRST_PCA9685_EXPANDER_ADDRESS);
    writeAllServos(0); 
    delay(500);
    
    stepper3.setMaxSpeed(1000.0);
    stepper1.setMaxSpeed(1000.0);
}

void loop() {
//    stepper1.moveTo(2024); 
//    stepper1.setSpeed(400.0);
//    stepper1.runSpeedToPosition();
//    
//    stepper3.moveTo(2024); 
//    stepper3.setSpeed(400.0);
//    stepper3.runSpeedToPosition();

    ServoEasing::ServoEasingArray[3]->easeTo(0,SERVO_SPEED);
    delay(1000);
    ServoEasing::ServoEasingArray[3]->easeTo(45,SERVO_SPEED);
    delay(1000);
    
}
