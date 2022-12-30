
// CNC Shield Stepper  Control Demo
// Superb Tech
// www.youtube.com/superbtech

const int StepX = 2;
const int DirX = 5;
const int StepY = 3;
const int DirY = 6;
const int StepZ = 4;
const int DirZ = 7;

void setup()
{
  pinMode(StepX, OUTPUT);
  pinMode(DirX, OUTPUT);
  pinMode(StepY, OUTPUT);
  pinMode(DirY, OUTPUT);
  pinMode(StepZ, OUTPUT);
  pinMode(DirZ, OUTPUT);
}

int full_steps = 200 * 20;
void loop()
{
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

  // digitalWrite(DirY, HIGH); // set direction, HIGH for CCW, LOW for CW
  // for (int x = 0; x < full_steps; x++)
  // { // loop for 200 steps
  //   digitalWrite(StepY, HIGH);
  //   delayMicroseconds(700);
  //   digitalWrite(StepY, LOW);
  //   delayMicroseconds(700);
  // }
  // delay(1000); // delay for 1 second

  // digitalWrite(DirY, LOW);
  // for (int x = 0; x < full_steps; x++)
  // { // loop for 200 steps
  //   digitalWrite(StepY, HIGH);
  //   delayMicroseconds(700);
  //   digitalWrite(StepY, LOW);
  //   delayMicroseconds(700);
  // }
  // delay(1000); // delay for 1 second

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
