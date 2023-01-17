#include <SerialTransfer.h>
#include <Stepper.h>

// Defines the number of steps per rotation
const int stepsPerRevolution = 2038;
const int delay_step = 2200;
const int num_steppers = 4;

// Creates an instance of stepper class
// Pins entered in sequence IN1-IN3-IN2-IN4 for proper step sequence
Stepper step1 = Stepper(stepsPerRevolution, 23, 27, 25, 29);
Stepper step2 = Stepper(stepsPerRevolution, 33, 37, 35, 39);
Stepper step3 = Stepper(stepsPerRevolution, 43, 47, 45, 49);
Stepper step4 = Stepper(stepsPerRevolution, 22, 26, 24, 28);
Stepper stepperList[4] = {step1, step2, step3, step4};

SerialTransfer myTransfer;
// put claw 45 degree - set program to think it's at 90
// claw is using the same stepper but it's designed to move CW
// for pos so dont need to change neg
long target_steps[4] = {0, 0, 0, 510};
long current_steps[4] = {0, 0, 0, 510};

void setup()
{
    Serial.begin(115200);
    myTransfer.begin(Serial);
}

void loop()
{
    if (myTransfer.available())
    {
        // send all received data back to Python
        for (uint16_t i = 0; i < myTransfer.bytesRead; i++)
        {
            myTransfer.packet.txBuff[i] = myTransfer.packet.rxBuff[i];
        }

        myTransfer.sendData(myTransfer.bytesRead);
        myTransfer.rxObj(target_steps);
    }

    // SWITCH DIR OUTPUT DEPENDING ON TARGET STEP
    // then move and update current step by inc or dec
      

    for (int n = 0; n < num_steppers; n++)
    {
        if (target_steps[n] > current_steps[n]) // CCW
        {
            stepperList[n].step(1);
            current_steps[n]++;
        }
        else if (target_steps[n] < current_steps[n]) // CW
        {
            stepperList[n].step(-1);
            current_steps[n]--;
        }
    }
    delayMicroseconds(delay_step);
}
