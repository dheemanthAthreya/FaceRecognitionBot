# Face Recognizer Bot

A computer vision based project that detects and recognizes the faces of people and stores the time of detection in a .csv file. This is based on the [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey. [Click here](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) to find out how it works.
Some of the applications of this project include:
- In schools, colleges, organizations and event venues to mark the attendance of people coming in.
- In home surveillance systems to raise an alert if an unknown person is approaching and track them (This is still a work in progress and will be implemented in the future).
- Using it with speech recognition in humanoid robots to personalize conversations.

In schools, this can be used with a database to maintain the attendance record of students and the .csv file can also store the place of detection in surveillance systems.

## Requirements:
- A working Python installation.
- A functioning webcam or a video feed
- CMake and a C compiler to build dlib
- dlib library
- face_recognition library
- OpenCV library ```pip install opencv-python ```
- numpy library ```pip install numpy ```

[Guide to install CMake and dlib](https://www.geeksforgeeks.org/how-to-install-dlib-library-for-python-in-windows-10/)  
[Guide to install face_recognition](https://www.geeksforgeeks.org/how-to-install-face-recognition-in-python-on-windows/): Method 1 should work if CMake and dlib are installed properly.


## Installation:
- If the requirements are met, clone the repository and run the Facial_Recognition.py.

## Motivation:
- I wanted to build a bot that uses the surveillance system and sends an alert if an unknown person is approaching my own house. At this stage, this cannot yet be achieved but I'm currently working on it.  

## Note:
- ```TrainedImages``` contains images of faces that the bot can recognize.
- ```TestImages``` can be used to see if the bot is working.
- If no changes are made, the code takes video input from the default webcam.
- A pre-recorded video was played on a screen with the webcam in front of it. This setup was used to show a demo of the project and it is saved as ```result.mp4```.
