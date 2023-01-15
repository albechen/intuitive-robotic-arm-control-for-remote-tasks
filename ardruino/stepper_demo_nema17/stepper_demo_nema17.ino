const int StepX = 2;
const int DirX = 5;
const int StepY = 3;
const int DirY = 6;
const int StepZ = 4;
const int DirZ = 7;

long calculate_target_steps(long stepper_steps, long target_degree)
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

const long yz_stepper_steps = 200L * 4 * 6 * 6;
const long x_stepper_steps = 200L *8 * 20;
long target_array[4][3] = {
    {
        0,
        45,
        -45,
    },
    {
        45,
        90,
        -90,
    },
    {
        -45,
        75,
        -70,
    },
    {
        0,
        0,
        0,
    },
};

long stepsPos_x = 0;
long stepsPos_y = 0;
long stepsPos_z = 0;

int dirInc_x = 1;
int dirInc_y = 1;
int dirInc_z = 1;
// set direction, HIGH for CCW, LOW for CW
void loop()
{
  if (Serial.available() == 0)
  {
    for (int target_num = 0; target_num < 4; ++target_num)
    {
      long stepsTarget_x = calculate_target_steps(x_stepper_steps, target_array[target_num][0]);
      long stepsToMove_x = stepsTarget_x - stepsPos_x;
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

      long stepsTarget_y = calculate_target_steps(yz_stepper_steps, target_array[target_num][1]);
      long stepsToMove_y = stepsTarget_y - stepsPos_y;
      if (stepsToMove_y > 0)
      {
        dirInc_y = 1;
        digitalWrite(DirY, HIGH); // CCW
      }
      else
      {
        dirInc_y = -1;
        digitalWrite(DirY, LOW); // CW
      }

      long stepsTarget_z = calculate_target_steps(yz_stepper_steps, target_array[target_num][2]);
      long stepsToMove_z = stepsTarget_z - stepsPos_z;

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

      long absStepsToMove_x = abs(stepsToMove_x);
      long absStepsToMove_y = abs(stepsToMove_y);
      long absStepsToMove_z = abs(stepsToMove_z);

      long max_move = max(max(absStepsToMove_x, absStepsToMove_y), absStepsToMove_z);

      for (int x = 0; x <= max_move; x++)
      { // loop for 200 steps
        if (x < absStepsToMove_x)
        {
          digitalWrite(StepX, HIGH);
        }
        if (x < absStepsToMove_y)
        {
          digitalWrite(StepY, HIGH);
        }
        if (x < absStepsToMove_z)
        {
          digitalWrite(StepZ, HIGH);
        }

        delayMicroseconds(600);

        

        if (x < absStepsToMove_x)
        {
          digitalWrite(StepX, LOW);
          stepsPos_x += dirInc_x;
        }
        if (x < absStepsToMove_y)
        {
          digitalWrite(StepY, LOW);
          stepsPos_y += dirInc_y;
        }
        if (x < absStepsToMove_z)
        {
          digitalWrite(StepZ, LOW);
          stepsPos_z += dirInc_z;
        }

        delayMicroseconds(600);
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
    long stepsTarget_x = calculate_target_steps(x_stepper_steps, 0);
    long stepsToMove_x = stepsTarget_x - stepsPos_x;
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

    long stepsTarget_y = calculate_target_steps(yz_stepper_steps, 0);
    long stepsToMove_y = stepsTarget_y - stepsPos_y;
    if (stepsToMove_y > 0)
    {
      dirInc_y = 1;
      digitalWrite(DirY, HIGH); // CCW
    }
    else
    {
      dirInc_y = -1;
      digitalWrite(DirY, LOW); // CW
    }

    long stepsTarget_z = calculate_target_steps(yz_stepper_steps, 0);
    long stepsToMove_z = stepsTarget_z - stepsPos_z;

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

    long absStepsToMove_x = abs(stepsToMove_x);
    long absStepsToMove_y = abs(stepsToMove_y);
    long absStepsToMove_z = abs(stepsToMove_z);

    long max_move = max(max(absStepsToMove_x, absStepsToMove_y), absStepsToMove_z);

    for (int x = 0; x <= max_move; x++)
    { // loop for 200 steps
      if (x < absStepsToMove_x)
      {
        digitalWrite(StepX, HIGH);
      }
      if (x < absStepsToMove_y)
      {
        digitalWrite(StepY, HIGH);
      }
      if (x < absStepsToMove_z)
      {
        digitalWrite(StepZ, HIGH);
      }

      delayMicroseconds(400);

      

      if (x < absStepsToMove_x)
      {
        digitalWrite(StepX, LOW);
        stepsPos_x += dirInc_x;
      }
      if (x < absStepsToMove_y)
      {
        digitalWrite(StepY, LOW);
        stepsPos_y += dirInc_y;
      }
      if (x < absStepsToMove_z)
      {
        digitalWrite(StepZ, LOW);
        stepsPos_z += dirInc_z;
      }

      delayMicroseconds(400);
    }

    Serial.print("ENDING");
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
