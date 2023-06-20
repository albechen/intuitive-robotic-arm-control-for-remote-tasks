# Intuitive Robotic Arm Control for Remote Tasks

With the rise in remote work, this project aims to support preforming in person tasks  remeotely. With just two webcams on the local user's side, they will be able to control a robotic arm with just their natraul hand movements by capturing and interpreting their hand's 3D location and gestures in real-time.

Intiially my ambitions were unrealistically high for this project, but the process allowed me to learn eletrical and mechnical concepts that I've never been exposed to.

    VIDEO OF ARM BEING CONTROLLED / PICKING UP STUFF 

## Overview

The goals of this project was to create a robotic arm that (1) can be controlled by the user remotely, (2) only requires the user to locally have two webcams and their hands, and (3) is dexterious enough to pick up simple objects.

The process for making the arm centered around the two main issues (1) how to detect the location and guestures of a hand with just webcams, and (2) how to make a robotic arm with a decent range of motion

![flowchart](/images/flowchart.jpg "flowchart")

**Tools and Skills Learned:** Python CV Libraries (OpenCV, MediaPipe), Arduino, 3D printing mechnaical parts, Fusion 360 designing, Stepper & Servos principles (gear boxes are great), Intro to not blowing yourself up using electronic

## Hand Detection and 3D Location

Hand detection was realtively easy to implement with MediaPipe's out of the box hand detection model. However, the output would be the 2D location of the hand on the screen -- which  would not account for the depth and innate properites of the webcam.

Therefore, there's where the mono and stero calibration of the two cameras came in. Intially monoclibrate each camera invdivually to get their intrenstic and distortiaton matrices to adjust any of the webcam's own offsets. Then stero calibration can get the rotational and transpose differences between the two cameara to understand their realtive posiitons. Additonally -- the setro calibration is also then adjusted to calribate for a specific world position of where to set (0, 0, 0) in our 3D space.

![hand_detection](/images/hand_detection_calib.jpg "hand_detection_calib")

Once these rotational and transpose matrices are calibrated - they can be combine into a projection matrix that summarizes how to adjust the pixel locations of each webcam's feed. With a bit of math, we're able to calculate the relative 3D location.

![hand_detection](/images/hand_detection_3d.gif "hand_detection_3d")

## Robotic Arm Design

The design of the robotic arm was the part of the project where I had the least epxerince in -- mechanial and eletrical engineering. THe main areas I had to focus on can be broken down into the following

1. Arm Mechanism
2. Joint Design
3. Arduino Implemntation
4. Electronic Considerations

**Arm Mechnaism**
The design choice was based off a 6 DOF robot commonly used in industry. There are so mnay resources surroudning the basis of this robot design and it's fairly simple. The arm can be broken down into two parts, the arm and wrist. The arm would bring the hand to the correct location and the wrist woudl subsequently move the hand to face the right direction.

**Joint Design**
A large portion of designin the arm centered around the weight consideration. ex: What part to use that has enough torque for the joints? How to design the arm to minize weight futher down the arm? How to optimize strength vs weight? How to design the movement to be smoother?

- **Version 1: Servos** -- Initally the project was soley going to use servos since they have their own internal potentiomenter, have high torque per weight, and are cheap. However, the main drawback is they dont provide smooth movement and dont adjust well to the live feed of incoming target angles
- **Version 2: Shoulder Design** -- With moving to steppers, the smoothness as sorted out, but the weight of the servos at the elbow and wrist were doubling the weight and they didn't even have as high of a toque as the servos. Therefore, I looked to move the weight of the steppers to the shoulder.
- **Version 3: Gear Boxes** -- While moveing the elbow stepper to the shoulder worked, the design limited movement and increased complexity. Therefore, the stepper could be kept that the elbow if gear boxes were used. A plantery gear box was designed to increase the torque of the steppers for the shoulder and elbow joints which allowed them to shoulder the weight of the arm.

![arm_design](/images/arm_design_version.gif "arm_design_version")

**Arduino Implmentation**
I have no exprince coding in C or C++ since freshman year of college for a single project so I was a bit out of my element. Without the luxuery of R's or Python's infinate libraries I was a bit out of my element, but quicly found it to be nice knowing expicently what functions I'm sending the arduino.

The main issues I found were trying to send streaming data of target angles to each of the seven steppers that I was controlling. Overall the logic wasn't too difficult, but maintaining and implenmting the mechanical limitations into the code was something I had to quickly pickup on.

**Electronic Considerations**
When working with the electronics I believe I was rightfully cautious of each component. Expcially when owrking with power supplies -- I found that powering seven servos woudl require more than a simple 5v 5A power supply. It was good to finllay dive into the basics of electronics, but throughout this project I was always scared of starting a fire or frying my entire proejct.

## Mechnaisal and Software Integration

When working with my projects in Chemical Enginnering or Data Science, there's alwasy some intregation issues, but especially when working in the totally new field of electric and mehcanical icompoentts I found myself floundering. I wont write out all the issues, but I was just going through integration hell.

**Remote Usage**
Using serail and socket were pleasently the most comformatable I felt in the final steps of the project. Since I was just testing through my local network there weren't too many issues. Ofcourse if this project were to truely be rotemoete there woudl be more qork reqired.

![integration](/images/integration_remote.gif "integration_remote")

**Inverse Kinamatics**
Inveverse kinamatics wasn't 100% integration, since it should just be math that works. However, I found myself watching through multiple online recorded seminars on robotics to implement it. The main issue was translating my design of the 6 DOF robot -> matrices that describe it -> feeding test data using the actual designed robot. At each of these steps I found bug and issues all round with my math, how I designed the robot vs. how the matrices describ it, how I assemebled it, and all sorts of fun.

## Final Results
