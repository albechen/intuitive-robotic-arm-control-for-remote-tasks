#include <AccelStepper.h>
#include <MultiStepper.h>

const int num_angles = 7;
int pos_list[num_angles];                          // this value is the upgratable data
byte *ddata = reinterpret_cast<byte *>(&pos_list); // pointer for transferData()
size_t pcDataLen = sizeof(pos_list);
bool newData = false;

// Define some steppers and the pins the will use
AccelStepper sm1(AccelStepper::FULL4WIRE, 23, 27, 25, 29);
AccelStepper sm2(AccelStepper::FULL4WIRE, 33, 37, 35, 39);
AccelStepper sm3(AccelStepper::FULL4WIRE, 43, 47, 45, 49);
AccelStepper sm4(AccelStepper::FULL4WIRE, 53, 57, 55, 59);

int calculate_target_steps(int steps_fullTurn, int target_degree)
{
    long large_num = (long)target_degree * (long)steps_fullTurn;
    long stepper_fraction = large_num / 3600;
    return lround(stepper_fraction);
}

const int steps_28BYJ = 2048;
const int steps_nema17_p = 200 * 2 * 6 * 6;
const int steps_nema17_c = 200 * 2 * 20;
MultiStepper steppers;

void setup()
{
    Serial.begin(115200);

    sm1.setMaxSpeed(200.0);
    sm1.setAcceleration(100.0);

    sm2.setMaxSpeed(200.0);
    sm2.setAcceleration(100.0);

    sm3.setMaxSpeed(200.0);
    sm3.setAcceleration(100.0);

    sm4.setMaxSpeed(200.0);
    sm4.setAcceleration(100.0);

    steppers.addStepper(sm1);
    steppers.addStepper(sm2);
    steppers.addStepper(sm3);
    steppers.addStepper(sm4);
}

void checkForNewData()
{
    if (Serial.available() >= pcDataLen && newData == false)
    {
        byte inByte;
        for (byte n = 0; n < pcDataLen; n++)
        {
            ddata[n] = Serial.read();
        }
        while (Serial.available() > 0)
        { // now make sure there is no other data in the buffer
            byte dumpByte = Serial.read();
            Serial.println(dumpByte);
        }
        newData = true;
    }
}

void loop()
{
    checkForNewData();
    if (newData == true)
    {
        newData = false;
    }

    stepper.moveTo(pos_list);
    stepper.run();

    // else
    // {
    //     for (int i=0, i<num_angles, i++){
    //         pos_list[i] = 0;
    //     }
    //     stepper.moveTo(pos_list);
    //     stepper.run();
    // }
}
