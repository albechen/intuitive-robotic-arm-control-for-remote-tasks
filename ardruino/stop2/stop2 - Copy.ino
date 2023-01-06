int myArray[15]; //this value is the upgratable data
byte* ddata = reinterpret_cast<byte*>(&myArray); // pointer for transferData()
size_t pcDataLen = sizeof(myArray);
bool newData=false;

void setup() {
    Serial.begin(115200);//baudrate
}

void loop() {
    checkForNewData();
    if (newData == true) {
        newData = false;
    }
    toPy(myArray[0],myArray[1],myArray[2],myArray[3],myArray[4],myArray[5],myArray[6],myArray[7],myArray[8],
    myArray[9],myArray[10],myArray[11],myArray[12],myArray[13],myArray[14],0,0,0,random(100)); //here write the send data
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


void toPy(int a,int b,int c,int d,int e,int f,
          int g,int h,int i,int j,int k,int l,
          int m,int n,int o,int p,int q,int r,int s)//19 datas
{
  //rpidata="[1,2,3,4,5,6,7,8,9,10,111,0]";
String data="["+String(a)+","+String(b)+","+String(c)+","+String(d)+","+String(e)+","+String(f)+","+String(g)+","+String(h)+","+String(i)+","+
String(j)+","+String(k)+","+String(l)+","+String(m)+","+String(n)+","+String(o)+","+String(p)+","+String(q)+","+String(r)+","+String(s)+"]";
  delay(50);Serial.println(data);
}
