const int StepX = 2;
const int DirX = 5;
const int StepY = 3;
const int DirY = 6;
const int StepZ = 4;
const int DirZ = 7;

int calculate_target_steps(int stepper_steps, int target_degree)
{
  long large_num = (long)target_degree * (long)stepper_steps;
  long stepper_fraction = large_num / 360;
  return lround(stepper_fraction);
}

void setup()
{
  Serial.begin(9600);
  pinMode(StepX, OUTPUT);
  pinMode(DirX, OUTPUT);
  pinMode(StepY, OUTPUT);
  pinMode(DirY, OUTPUT);
  pinMode(StepZ, OUTPUT);
  pinMode(DirZ, OUTPUT);
}

const int stepper_steps = 200 * 6;
int target_array[4][2] = {
    {
        0,
        20,
    },
    {
        90,
        20,
    },
    {
        90,
        90,
    },
    {
        45,
        45,
    },
};

int stepsPos_x = 0;
int stepsPos_z = calculate_target_steps(stepper_steps, 20);

int dirInc_x = 1;
int dirInc_z = 1;
// set direction, HIGH for CCW, LOW for CW
void loop()
{
  if (Serial.available() == 0)
  {
    for (int target_num = 0; target_num < 4; ++target_num)
    {
      int stepsTarget_x = calculate_target_steps(stepper_steps, target_array[target_num][0]);
      int stepsToMove_x = stepsTarget_x - stepsPos_x;
      if (stepsToMove_x > 0)
      {
        dirInc_x = 1;
        digitalWrite(DirX, HIGH); // CCW
      }
      else
      {
        dirInc_x = -1;
        digitalWrite(DirX, LOW); // CW
      }

      int stepsTarget_z = calculate_target_steps(stepper_steps, target_array[target_num][1]);
      int stepsToMove_z = stepsTarget_z - stepsPos_z;

      if (stepsToMove_z > 0)
      {
        dirInc_z = 1;
        digitalWrite(DirZ, LOW); // CW
      }
      else
      {
        dirInc_z = -1;
        digitalWrite(DirZ, HIGH); // CCW
      }

      int absStepsToMove_x = abs(stepsToMove_x);
      int absStepsToMove_z = abs(stepsToMove_z);

      int max_move = max(absStepsToMove_x, absStepsToMove_z);

      for (int x = 0; x <= max_move; x++)
      { // loop for 200 steps
        if (x < absStepsToMove_x)
        {
          digitalWrite(StepX, HIGH);
        }
        if (x < absStepsToMove_z)
        {
          digitalWrite(StepZ, HIGH);
        }

        delayMicroseconds(700);

        if (x < absStepsToMove_x)
        {
          digitalWrite(StepX, LOW);
          stepsPos_x += dirInc_x;
        }
        if (x < absStepsToMove_z)
        {
          digitalWrite(StepZ, LOW);
          stepsPos_z += dirInc_z;
        }

        delayMicroseconds(700);
      }

      Serial.print(target_num);
      Serial.print("\t");
      Serial.print(max_move);
      Serial.print("\t");
      Serial.print("\t");
      Serial.print("\t");
      Serial.print(stepsTarget_x);
      Serial.print("\t");
      Serial.print(stepsToMove_x);
      Serial.print("\t");
      Serial.print(dirInc_x);
      Serial.print("\t");
      Serial.print(stepsPos_x);
      Serial.print("\t");
      Serial.print("\t");
      Serial.print("\t");
      Serial.print(stepsTarget_z);
      Serial.print("\t");
      Serial.print(stepsToMove_z);
      Serial.print("\t");
      Serial.print(dirInc_z);
      Serial.print("\t");
      Serial.print(stepsPos_z);
      Serial.print("\n");

      delay(1000);
    }
  }

  else
  {
    int stepsTarget_x = calculate_target_steps(stepper_steps, 0);
    int stepsToMove_x = stepsTarget_x - stepsPos_x;
    if (stepsToMove_x > 0)
    {
      dirInc_x = 1;
      digitalWrite(DirX, HIGH); // CCW
    }
    else
    {
      dirInc_x = -1;
      digitalWrite(DirX, LOW); // CW
    }

    int stepsTarget_z = calculate_target_steps(stepper_steps, 20);
    int stepsToMove_z = stepsTarget_z - stepsPos_z;

    if (stepsToMove_z > 0)
    {
      dirInc_z = 1;
      digitalWrite(DirZ, LOW); // CW
    }
    else
    {
      dirInc_z = -1;
      digitalWrite(DirZ, HIGH); // CCW
    }

    int absStepsToMove_x = abs(stepsToMove_x);
    int absStepsToMove_z = abs(stepsToMove_z);

    int max_move = max(absStepsToMove_x, absStepsToMove_z);

    for (int x = 0; x <= max_move; x++)
    { // loop for 200 steps
      if (x < absStepsToMove_x)
      {
        digitalWrite(StepX, HIGH);
      }
      if (x < absStepsToMove_z)
      {
        digitalWrite(StepZ, HIGH);
      }

      delayMicroseconds(700);

      if (x < absStepsToMove_x)
      {
        digitalWrite(StepX, LOW);
        stepsPos_x += dirInc_x;
      }
      if (x < absStepsToMove_z)
      {
        digitalWrite(StepZ, LOW);
        stepsPos_z += dirInc_z;
      }

      delayMicroseconds(700);
    }

    Serial.print('ENDING');
    Serial.print("\t");
    Serial.print(max_move);
    Serial.print("\t");
    Serial.print("\t");
    Serial.print("\t");
    Serial.print(stepsTarget_x);
    Serial.print("\t");
    Serial.print(stepsToMove_x);
    Serial.print("\t");
    Serial.print(dirInc_x);
    Serial.print("\t");
    Serial.print(stepsPos_x);
    Serial.print("\t");
    Serial.print("\t");
    Serial.print("\t");
    Serial.print(stepsTarget_z);
    Serial.print("\t");
    Serial.print(stepsToMove_z);
    Serial.print("\t");
    Serial.print(dirInc_z);
    Serial.print("\t");
    Serial.print(stepsPos_z);
    Serial.print("\n");

    delay(1000);
    exit(0);
  }
}