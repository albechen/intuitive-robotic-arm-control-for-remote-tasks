const int num_angles = 5;
String incoming[num_angles];
int int_list[num_angles];

String raw_incoming;
char temp_int;
int count = 0;

void setup()
{
  Serial.begin(9600);
}

void loop()
{
  while (Serial.available() > 0)
  {
    raw_incoming = Serial.readString();
    int str_len = raw_incoming.length() + 1;
    char char_array[str_len];
    raw_incoming.toCharArray(char_array, str_len);

    for(int i =0; i < strlen(char_array); i++ ) {
      char c = char_array[i];
      count = 0;
      if (c == "$"){
        incoming[count] = temp_int;
        int_list[count] = incoming[count].toInt();
        count += 1;
      }
      else{
        temp_int += c;
      }
    }

    
    for (int i = 0; i < num_angles; i++)
    {
      Serial.print(incoming[i].toInt());
    }
    
    Serial.print(".......");
    for (int i = 0; i < num_angles; i++)
    {
      Serial.print(incoming[i]);
    }
    
    Serial.print("\n");
  }
}
