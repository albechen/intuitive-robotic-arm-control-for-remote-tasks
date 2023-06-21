# Intuitive Robotic Arm Control for Remote Tasks

With the rise in remote work, this project aims to support preforming in person tasks  remotely. With just two webcams on the local user's side, they will be able to control a robotic arm with just their natural hand movements by capturing and interpreting their hand's 3D location and gestures in real-time.

<img src="/images/final_red_green.gif?raw=true" width="1000px">

## Overview

The goals of this project were to create a robotic arm that (1) can be controlled by the user remotely, (2) only requires the user to locally have two webcams and their hands, and (3) is dexterous enough to pick up simple objects.

**Skill learned during project:**

1. **Computer Vision** applications in physical world
   - MediaPipe and OpenCV, DLT and triangulation
2. Working with **electrical** components
   - Arduino, coding in C++, how to not fry myself on a power supply, steppers, servos
3. Applying basic **robotics** principles and math behind them
   - 6 DOF design, inverse and forward kinematics, gear box designing, fusion 360
4. **Designing** a physical product and **integrating** software surrounding it

The process for making the arm centered around the two main issues (1) how to detect the location and gestures of a hand with just webcams, and (2) how to make a robotic arm with a decent range of motion.

![flowchart](/images/flowchart.jpg "flowchart")

## Hand Detection and 3D Location

Hand detection was relatively easy to implement with MediaPipe's out of the box hand detection model. However, the output would be the 2D location of the hand on the screen -- which  would not account for the depth and innate properties of the webcam.

Therefore, there's where the mono and stereo calibration of the two cameras came in. Initially mono calibration each camera individually to get their intrinsic and distortion matrices to adjust any of the webcam's own offsets. Then stereo calibration can get the rotational and transpose differences between the two cameras to understand their relative positions. Additionally -- the stereo calibration is also then adjusted to calibrate for a specific world position of where to set (0, 0, 0) in our 3D space.

![hand_detection](/images/hand_detection_calib.jpg "hand_detection_calib")

Once these rotational and transpose matrices are calibrated - they can combine into a projection matrix that summarizes how to adjust the pixel locations of each webcam's feed. With a bit of math, we're able to calculate the relative 3D location.

<img src="/images/hand_detection_3d.gif?raw=true" width="1000px">

## Robotic Arm Design

The design of the robotic arm was the part of the project where I had the least experience in -- mechanical and electrical engineering. The main areas I had to focus on can be broken down into the following.

1. Arm Mechanism
2. Joint Design
3. Arduino Implementation
4. Electronic Considerations

**Arm Mechanism**
The design choice was based off a 6 DOF robot commonly used in industry. There are so many resources surrounding the basis of this robot design and it's fairly simple. The arm can be broken down into two parts, the arm and wrist. The arm would bring the hand to the correct location and the wrist would subsequently move the hand to face the right direction.

**Joint Design**
A large portion of designing the arm centered around the weight consideration. ex: What part to use that has enough torque for the joints? How to design the arm to minimize weight further down the arm? How to optimize strength vs weight? How to design the movement to be smoother?

- **Version 1: Servos** -- Initially the project was solely going to use servos since they have their own internal potentiometer, have high torque per weight, and are cheap. However, the main drawback is they don’t provide smooth movement and don’t adjust well to the live feed of incoming target angles.
- **Version 2: Shoulder Design** -- With moving to steppers, the smoothness as sorted out, but the weight of the servos at the elbow and wrist were doubling the weight and they didn't even have as high of a toque as the servos. Therefore, I looked to move the weight of the steppers to the shoulder.
- **Version 3: Gear Boxes** -- While moving the elbow stepper to the shoulder worked, the design limited movement and increased complexity. Therefore, the stepper could be kept that the elbow if gear boxes were used. A planetary gear box was designed to increase the torque of the steppers for the shoulder and elbow joints which allowed them to shoulder the weight of the arm.

<img src="/images/arm_design_version.gif?raw=true" width="1000px">

**Arduino Implementation**
I have no experience coding in C or C++ since freshman year of college for a single project, so I was a bit out of my element. Without the luxury of R's or Python's infinite libraries I was a bit out of my element, but quickly found it to be nice knowing explicitly what functions I'm sending the Arduino.

The main issues I found were trying to send streaming data of target angles to each of the seven steppers that I was controlling. Overall, the logic wasn't too difficult, but maintaining and implementing the mechanical limitations of the code was something I had to quickly pickup on.

**Electronic Considerations**
When working with electronics I believe I was rightfully cautious of each component. Especially when working with power supplies -- I found that powering seven servos would require more than a simple 5v 5A power supply. It was good to finally dive into the basics of electronics, but throughout this project I was always scared of starting a fire or frying my entire project.

## Mechanical and Software Integration

When working with my projects in Chemical Engineering or Data Science, there's always some integration issues, but especially when working in the totally new field of electric and mechanical components I found myself floundering. I won’t write out all the issues, but I was just going through integration hell.

**Remote Usage**
Using serial and socket were pleasantly the most comfortable I felt in the final steps of the project. Since I was just testing through my local network there weren't too many issues. Of course, if this project were to truly be remote there would be more work required.

<img src="/images/integration_remote.gif?raw=true" width="1000px">

**Inverse Kinematics**
Inverse kinematics wasn't 100% integration, since it should just be math that works. However, I found myself watching multiple online recorded seminars on robotics to implement it. The main issue was translating my design of the 6 DOF robot -> matrices that describe it -> feeding test data using the actual designed robot. At each of these steps I found bugs and issues all round with my math, how I designed the robot vs. how the matrices describe it, how I assembled it, and all sorts of fun.

## Conclusion

While the final product is by no means ready for any real applications, it was a great exercise in learning new skills. The project was mainly limited by (1) the accuracy of the cameras and the resulting location of my hand to flicker which caused stutters in the arm and (2) the lack of homing of the steppers. It would be interesting to pursue coding solutions to problem 1, but overall, I'm happy with the progress!

I greatly valued learning more about practical applications and integration of software with  physical products. It was a great jumping off point for more electrically or mechanically intensive projects I'm planning on in the future!

Also -- now I have an arm that can do this:

<img src="/images/final_bread.gif?raw=true" width="1000px">
