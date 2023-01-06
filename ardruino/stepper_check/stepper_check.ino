#include <SerialTransfer.h>

SerialTransfer myTransfer;

const int STEP_X = 2;
const int DIR_X = 5;
const int STEP_Y = 3;
const int DIR_Y = 6;
const int STEP_Z = 4;
const int DIR_Z = 7;

const int STEP_list[3] = {STEP_X, STEP_Y, STEP_Z};
const int DIR_list[3] = {DIR_X, DIR_Y, DIR_Z};

const int num_steppers = 3;
long target_steps[3] = {0, 0, 0};
long current_steps[3] = {0, 0, 0};

// set direction, HIGH for CCW, LOW for CW

void setup()
{
  Serial.begin(115200);
  myTransfer.begin(Serial);

  pinMode(STEP_X, OUTPUT);
  pinMode(DIR_X, OUTPUT);
  pinMode(STEP_Y, OUTPUT);
  pinMode(DIR_Y, OUTPUT);
  pinMode(STEP_Z, OUTPUT);
  pinMode(DIR_Z, OUTPUT);
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
    if (target_steps[n] > current_steps[n])
    {
      digitalWrite(DIR_list[n], HIGH); // CCW
      current_steps[n]++;
    }
    else if (target_steps[n] < current_steps[n]) // CW
    {
      digitalWrite(DIR_list[n], LOW); // CW
      current_steps[n]--;
    }
  }

  for (int n = 0; n < num_steppers; n++)
  {
    if (target_steps[n] != current_steps[n])
    {
      digitalWrite(STEP_list[n], HIGH);
    }
  }
  delayMicroseconds(500);

  for (int n = 0; n < num_steppers; n++)
  {
    if (target_steps[n] != current_steps[n])
    {
      digitalWrite(STEP_list[n], LOW);
    }
  }
  delayMicroseconds(500);
}
