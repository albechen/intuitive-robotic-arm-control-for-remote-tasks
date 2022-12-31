const int num_angles = 5;
int myArray[num_angles]; //this value is the upgratable data
byte* ddata = reinterpret_cast<byte*>(&myArray); // pointer for transferData()
size_t pcDataLen = sizeof(myArray);
bool newData=false;

void setup() {
    Serial.begin(9600);//baudrate
}

void loop() {
    checkForNewData();
    if (newData == true) {
        newData = false;
    }
    toPy(myArray[0],myArray[1],myArray[2],myArray[3],myArray[4]); //here write the send data
    }

void checkForNewData () {
    if (Serial.available() >= pcDataLen && newData == false) {
        byte inByte;
        for (byte n = 0; n < pcDataLen; n++) {
            ddata [n] = Serial.read();
        }
        while (Serial.available() > 0) { // now make sure there is no other data in the buffer
             byte dumpByte =  Serial.read();
             Serial.println(dumpByte);
        }
        newData = true;
    }
}


void toPy(int a,int b,int c,int d,int e)//19 datas
{
  //rpidata="[1,2,3,4,5,6,7,8,9,10,111,0]";
String data="["+String(a)+","+String(b)+","+String(c)+","+String(d)+","+String(e)+"]";
  delay(50);Serial.println(data);
}
