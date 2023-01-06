#include <AccelStepper.h>
#include <MultiStepper.h>
#include <SerialTransfer.h>

SerialTransfer myTransfer;

int pos_list[3];

const int STEP_X = 2;
const int DIR_X = 5;
const int STEP_Y = 3;
const int DIR_Y = 6;
const int STEP_Z = 4;
const int DIR_Z = 7;

AccelStepper bg1(AccelStepper::DRIVER, STEP_X, DIR_X);
AccelStepper bg2(AccelStepper::DRIVER, STEP_Y, DIR_Y);
AccelStepper bg3(AccelStepper::DRIVER, STEP_Z, DIR_Z);

MultiStepper steppers;

void setup()
{
    Serial.begin(115200);
    myTransfer.begin(Serial);

    bg1.setMaxSpeed(3000.0);
    bg1.setSpeed(2000);
    bg1.moveTo(0);

    bg2.setMaxSpeed(3000.0);
    bg2.setSpeed(2000);
    bg2.moveTo(0);

    bg3.setMaxSpeed(3000.0);
    bg3.setSpeed(2000);
    bg3.moveTo(0);

    // steppers.addStepper(sm1);
    // steppers.addStepper(sm2);
    // steppers.addStepper(sm3);
}


void loop()
{
    if(myTransfer.available())
  {
    // send all received data back to Python
    for(uint16_t i=0; i < myTransfer.bytesRead; i++)
      myTransfer.packet.txBuff[i] = myTransfer.packet.rxBuff[i];
    
    myTransfer.sendData(myTransfer.bytesRead);

    myTransfer.rxObj(pos_list);
  }


    bg1.moveTo(pos_list[0]);
    bg1.runSpeedToPosition();

    bg2.moveTo(pos_list[1]);
    bg2.runSpeedToPosition();

    bg3.moveTo(pos_list[2]);
    bg3.runSpeedToPosition();
}
