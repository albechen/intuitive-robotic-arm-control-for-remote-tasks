#include <AccelStepper.h>

// Define some steppers and the pins the will use
AccelStepper stepper1(AccelStepper::FULL4WIRE, 23, 27, 25, 29);
AccelStepper stepper2(AccelStepper::FULL4WIRE, 33, 37, 35, 39);
AccelStepper stepper3(AccelStepper::FULL4WIRE, 43, 47, 45, 49);

int calculate_target_steps(int stepper_steps, int target_degree)
{
    long large_num = (long)target_degree * (long)stepper_steps;
    long stepper_fraction = large_num / 360;
    return lround(stepper_fraction);
}

const int stepper_steps = 2048;
int target_array[3][3] = {{0, 0, 0,},
                       { 90, 90, 90,},
                       { 135, -45, 45,}};

//                       {0, 0, 0,},
int target_count = 0;

void setup()
{
    Serial.begin(9600);
    stepper1.setMaxSpeed(1000.0);
    stepper2.setMaxSpeed(1000.0);
    stepper3.setMaxSpeed(1000.0);
}

void loop()
{
  
  if(Serial.available() > 0){
    stepper1.moveTo(calculate_target_steps(stepper_steps, target_array[0][0])); 
    stepper1.setSpeed(400.0);
    stepper1.runSpeedToPosition();
  
    stepper2.moveTo(calculate_target_steps(stepper_steps, target_array[0][1])); 
    stepper2.setSpeed(400.0);
    stepper2.runSpeedToPosition();
    
    stepper3.moveTo(calculate_target_steps(stepper_steps, target_array[0][2])); 
    stepper3.setSpeed(400.0);
    stepper3.runSpeedToPosition();
  }

  else{
    stepper1.moveTo(calculate_target_steps(stepper_steps, target_array[target_count][0])); 
    stepper1.setSpeed(400.0);
    stepper1.runSpeedToPosition();
  
    stepper2.moveTo(calculate_target_steps(stepper_steps, target_array[target_count][1])); 
    stepper2.setSpeed(400.0);
    stepper2.runSpeedToPosition();
    
    stepper3.moveTo(calculate_target_steps(stepper_steps, target_array[target_count][2])); 
    stepper3.setSpeed(400.0);
    stepper3.runSpeedToPosition();

    if (stepper1.distanceToGo() == 0 && stepper2.distanceToGo() == 0 && stepper3.distanceToGo() == 0){

      delay(500);
      
      if (target_count == 2) {
        target_count = 0;
      }
      else {
        target_count += 1;
      }
    }
  }
}
