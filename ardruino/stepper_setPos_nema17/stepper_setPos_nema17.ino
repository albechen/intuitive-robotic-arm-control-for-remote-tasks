#include <AccelStepper.h>
#include <MultiStepper.h>

const int STEP_X = 2;
const int DIR_X = 5;
const int STEP_Y = 3;
const int DIR_Y = 6;
const int STEP_Z = 4;
const int DIR_Z = 7;

AccelStepper bg1(AccelStepper::DRIVER, STEP_X, DIR_X);
AccelStepper bg2(AccelStepper::DRIVER, STEP_Y, DIR_Y);
AccelStepper bg3(AccelStepper::DRIVER, STEP_Z, DIR_Z);

const int steps_28BYJ = 2048;
const int steps_nema17_p = 200 * 4 * 6 * 6;
const int steps_nema17_c = 200 * 8 * 20;

MultiStepper steppers;

void setup()
{
  pinMode(STEP_X, OUTPUT);
  pinMode(DIR_X, OUTPUT);
  pinMode(STEP_Y, OUTPUT);
  pinMode(DIR_Y, OUTPUT);
  pinMode(STEP_Z, OUTPUT);
  pinMode(DIR_Z, OUTPUT);
  
//    bg1.setMaxSpeed(1000.0);
//    bg1.setSpeed(1000);
//    bg1.moveTo(0);
//
    bg2.setMaxSpeed(2000);
    bg2.setAcceleration(2000);
    bg2.setSpeed(2000);
    bg2.moveTo(0);

    bg3.setMaxSpeed(1000);
    bg3.setAcceleration(2000);
    bg3.setSpeed(1000);
    bg3.moveTo(0);

//    steppers.addStepper(sm1);
//    steppers.addStepper(sm2);
//    steppers.addStepper(sm3);
}

void loop()
{   
//    bg2.moveTo(700);
//    bg2.run();


    bg3.moveTo(500);
    bg3.run();
    
  // digitalWrite(DirX, HIGH); // set direction, HIGH for CCW, LOW for CW
  // for (int x = 0; x < full_steps; x++)
  // { // loop for 200 steps
  //   digitalWrite(StepX, HIGH);
  //   delayMicroseconds(700);
  //   digitalWrite(StepX, LOW);
  //   delayMicroseconds(700);
  // }
  // delay(1000); // delay for 1 second

  // digitalWrite(DirX, LOW);
  // for (int x = 0; x < full_steps; x++)
  // { // loop for 200 steps
  //   digitalWrite(StepX, HIGH);
  //   delayMicroseconds(700);
  //   digitalWrite(StepX, LOW);
  //   delayMicroseconds(700);
  // }
  // delay(1000); // delay for 1 second

//   digitalWrite(DIR_Y, HIGH); // set direction, HIGH for CCW, LOW for CW
//   for (int x = 0; x < 700; x++)
//   { // loop for 200 steps
//     digitalWrite(STEP_Y, HIGH);
//     delayMicroseconds(700);
//     digitalWrite(STEP_Y, LOW);
//     delayMicroseconds(700);
//   }
//   delay(1000); // delay for 1 second
//
//   digitalWrite(DIR_Y, LOW);
//   for (int x = 0; x < 700; x++)
//   { // loop for 200 steps
//     digitalWrite(STEP_Y, HIGH);
//     delayMicroseconds(700);
//     digitalWrite(STEP_Y, LOW);
//     delayMicroseconds(700);
//   }
//   delay(1000); // delay for 1 second

  // digitalWrite(DirZ, HIGH);
  // for (int x = 0; x < full_steps; x++)
  // { // loop for 200 steps
  //   digitalWrite(StepZ, HIGH);
  //   delayMicroseconds(700);
  //   digitalWrite(StepZ, LOW);
  //   delayMicroseconds(700);
  // }
  // delay(1000); // delay for 1 second
  // digitalWrite(DirZ, LOW);

  // for (int x = 0; x < full_steps; x++)
  // { // loop for 200 steps
  //   digitalWrite(StepZ, HIGH);
  //   delayMicroseconds(700);
  //   digitalWrite(StepZ, LOW);
  //   delayMicroseconds(700);
  // }
  // delay(1000); // delay for 1 second
}
