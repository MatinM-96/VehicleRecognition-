Vehicle Detection and Classification System
===========================================
Overview
--------
This project implements a vehicle detection and classification system using Python and OpenCV. It is designed to 
process video, identify vehicles, classify them into categories (passenger car, van and motorcycle),
and manage parking space availability based on the vehicle type.



Features
------------
- **Real-time Vehicle Detection:** Detects moving vehicles in video frames using background subtraction and contour detection.
- **Vehicle Classification:** Classifies vehicles based on their dimensions into predefined categories.
- **Parking Space Management:** Monitors and updates the availability of parking spaces for different types of vehicles.
- **User Interface:** Displays processed video with vehicle detection and available parking spaces.


Prerequisites
-------------
    Python 3.8
    OpenCV-Python package
    NumPy package



Installation
------------
Ensure Python 3.8 is installed on your system.
Install OpenCV and NumPy using pip:

```angular2html
$ pip install opencv-python numpy
```
    

Usage
-----
Place your video file in the project directory.
Update the **videoPath** variable in the script with your video file's name.
Run the script in a Python environment:
````angular2html
$ python vehicle_detection.py
````
Interact with the program through the console to select the detection mode.


Functions Overview
------------------
- **centerHandle**: Calculates the center of a detected object.
- **classifyVehicles**: Classifies vehicles based on width.
- **processFrame**: Applies filters to enhance image quality for detection.
- **detectVehicles**: Identifies vehicles in the frame.
- **VehicleRecognitionEntry** and **VehicleRecognitionExit**: Core functions for vehicle recognition and classification.
    

Problems with the System
------------------------
1. Failure to Detect Bright and White Cars:
   -
    The system fail to detect some bright and white-colored cars.
    In order to show that the detection of vans is working, we set the width of the vans to be between 400 and 600 pixels from the center of the frame.
    This is only to demonstrate that it works for vans as well, and can be modified later. We did not capture any passing motorcycles either. The values for
    capturing the different types of vehicles can be modified in the **classifyVehicles(w)** function.
    
