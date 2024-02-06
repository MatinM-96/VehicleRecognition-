import cv2
import numpy as np



# this will calculate the center of the object
def centerHandle(x, y, w, h):

    x1 = int(w / 2)#calculate the center of the width
    y1 = int(h / 2)#calculate the center of the height
    cx = x + x1 #calculate the center of the x
    cy = y + y1 #calculate the center of the y
    return cx, cy #return the center of the x and y


def classifyVehicles(w):
    #it can be change to the weight of the vehicles, depending on the camera

    # if the width of the object is between 300 and 400, then it is a passenger car
    if 300 < w < 400:
        return "personbil"
    # if the width of the object is between 400 and 600, then it is a truck
    elif 400 <=w < 600:
        return "varebil"
    # if the width of the object is between 94 and 200, then it is a motorcycle
    elif 94 < w < 200:
        return "motorcykel"

    else:
        return "unknown"





#the function processFrame is used to process the frame and get a better image with less noise, and black and white image
#which it make easier to detect the vehicles
def processFrame(frame):
    frame = cv2.bilateralFilter(frame, 9, 75, 75) # Bilateral filter for noise reduction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #we get the frame in black and white
    blurred = cv2.GaussianBlur(gray, (3, 3), 5) #we apply a Gaussian filter to the frame to get rid of the dots
    return blurred





#this will return a list of  a point,
def detectVehicles(frame, DetectMOB, minWithReact, minHighetReact, count_line_x_position, offset):


    # we apply the background subtraction, to focus on the moving objects, and ignore other tings that not moving
    MovingObject = DetectMOB.apply(frame)

    # make bigger the moving objects, to make sure we didn't miss any object
    dilatedMovingObjects  = cv2.dilate(MovingObject, np.ones((5, 5)))

    morphologyKernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    #clean the  shapes on the moving objects
    refinedMotionShapes   = cv2.morphologyEx(dilatedMovingObjects , cv2.MORPH_CLOSE, morphologyKernel )
    refinedMotionShapes   = cv2.morphologyEx(refinedMotionShapes  , cv2.MORPH_CLOSE, morphologyKernel )


    #find the contours of the moving objects
    contours, _ = cv2.findContours(refinedMotionShapes  , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


#this function will display the number of available parking space when the vehicles enter the parking space
def DisplayAvailableParkingSpaceEntry(frame, passengerCar, motorCycle, van):


    if passengerCar == 0:
        cv2.putText(frame, f'No available parking for Passenger car', (500, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f'available parking for Passenger car: {passengerCar}', (500, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if van == 0:
        cv2.putText(frame, f'No available parking for van', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f'available parking for van: {van}', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if motorCycle == 0:
        cv2.putText(frame, f'No available parking for Motorcycle', (500, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f'available for Motorcycle: {motorCycle}', (500, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



#this function will display the number of available parking space when the vehicles exit the parking space
def DisplayAvailableParkingSpaceExit(frame, passengerCar, motorCycle, van):

    if  passengerCar == 0:
        cv2.putText(frame, f'No available parking for Passenger car', (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f'available parking for Passenger car: {passengerCar}', (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if van == 0:
        cv2.putText(frame, f'No available parking for van', (500, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f'available parking for van: {van}', (500, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if motorCycle == 0:
        cv2.putText(frame, f'No available parking for Motorcycle', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f'available for Motorcycle: {motorCycle}', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)








#this function will detect the vehicles that enter the parking space
def VehicleRecognitionEntry(video, passengerCar, motorCycle, truck):
    DetectMOB = cv2.createBackgroundSubtractorKNN(history=200, detectShadows=False)
    cap = cv2.VideoCapture(video)


    #if the video is not open, then print an error message
    if not cap.isOpened():
        print(f"Error opening video : {video}")
        return




    CounterPBil = 0
    CounterVBil = 0
    counterMCykel = 0
    CountTheNumberOfVehicles = 0



    # minimum width of the object
    minWithReact = 100

    # minimum height of the object
    minHighetReact = 60



    detect = []
    offset = 5  # allowable error between pixel
    count_line_x_position = 200

    while cap.isOpened(): #while the video is open
        ret, frame = cap.read() #read the video
        if not ret:        #if the video is not read
            print("Can't receive frame. Exiting ...")
            break

        # process the frame
        processedFrame = processFrame(frame)



        #detect the vehicles in the frame
        contours = detectVehicles(processedFrame, DetectMOB, minWithReact, minHighetReact, count_line_x_position, offset)

        #draw a line
        cv2.line(frame, (count_line_x_position, 0), (count_line_x_position, frame.shape[0]), (255, 127, 0), 3)




        #loop through the contours
        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            validateCounter = (w >= minWithReact) and (h >= minHighetReact)

            #if the width and height of the object is less than the minimum width and height, then continue
            if not validateCounter:
                continue

            #draw a rectangle around the vehicles
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)


            #calculate the center of the object
            center = centerHandle(x, y, w, h)
            #add the center of the object to the list
            detect.append(center)

            #loop through the list of the center of the object
            for (x, y) in detect:
                if count_line_x_position - offset < x < count_line_x_position + offset:
                    vehicleType = classifyVehicles(w)
                    CountTheNumberOfVehicles += 1
                    print("Count the amount of the vehicles: ", CountTheNumberOfVehicles)

                    if vehicleType == "personbil":
                        if(CounterPBil != passengerCar):
                            passengerCar -= 1

                        else:
                            print("Not available parking space for passenger car", CounterPBil)

                    elif vehicleType == "varebil":
                        if (CounterVBil != truck):
                            truck -= 1
                        else:
                            print("Not available parking space for van ", CounterVBil)
                    elif vehicleType == "motorcykel":
                        if (motorCycle != motorCycle):
                            motorCycle -= 1
                        else:
                            print("Not available parking space for motorcycle", motorCycle)
                    else:
                        print("vehicle are not allowed")
                    detect.remove((x, y)) #remove the center of the object from the list to avoid duplicate counting




        cv2.putText(frame, f'Vehicle enters the parking ', (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)


        #Display the number of available parking space
        DisplayAvailableParkingSpaceEntry(frame, passengerCar, motorCycle, truck)

        #show the video
        cv2.imshow('Car Detection', frame)
        #if the user press q, then exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    #print the total number of vehicles entered in the parking place
    print(" Total number of vehicles entered in the parking place: ", CountTheNumberOfVehicles)
    cap.release()
    cv2.destroyAllWindows()






def VehicleRecognitionExit(video, passengerCar, motorCycle, truck):
    DetectMOB = cv2.createBackgroundSubtractorKNN(history=200, detectShadows=False)
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print(f"Error opening video : {video}")
        return

    CountLevingv = 0

    minWithReact = 100
    minHighetReact = 60
    detect = []
    offset = 5  # allowable error between pixel
    countlLinexPosition = 200

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processedFrame = processFrame(frame)
        contours = detectVehicles(processedFrame, DetectMOB, minWithReact, minHighetReact, countlLinexPosition, offset)

        cv2.line(frame, (countlLinexPosition, 0), (countlLinexPosition, frame.shape[0]), (255, 127, 0), 3)

        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            validateCounter = (w >= minWithReact) and (h >= minHighetReact)
            if not validateCounter:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
            center = centerHandle(x, y, w, h)
            detect.append(center)

            for (x, y) in detect:
                if countlLinexPosition - offset < x < countlLinexPosition + offset:
                    vehicleType = classifyVehicles(w)
                    CountLevingv += 1
                    if vehicleType == "personbil":
                        passengerCar += 1
                    elif vehicleType == "varebil":
                            truck += 1
                    elif vehicleType == "motorcykel":
                            motorCycle += 1

                    detect.remove((x, y))



        cv2.putText(frame, f'Vehicles leaving the parking ', (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        #Display the number of available parking space
        DisplayAvailableParkingSpaceExit(frame, passengerCar, motorCycle, truck)

        cv2.imshow('Car Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Total number of vehicles leaving the parking place: ", CountLevingv)

    cap.release()
    cv2.destroyAllWindows()









def App():

    videoPath = 'carsData.mp4'


    #available parking space
    passengerCar = 6
    motorCycle = 2
    truck = 10

    passengerCarE = 0
    motorCycleE = 0
    truckE = 0

    #user input
    userInput = input( "Menu: \n"
                       "1. Vehicle Recognition for entering vehicles \n"
                       "2. Vehicle Recognition for exiting vehicles \n"
                       "Enter 'q' or '1' to exit app \n\n"
                       "Enter your choice:  ")

    if userInput == "1":
        VehicleRecognitionEntry(videoPath, passengerCar, motorCycle, truck)
    elif userInput == "2":
        VehicleRecognitionExit(videoPath, passengerCarE, motorCycleE, truckE)
    else:
        print("Goodbye, you enter the wrong input")
        exit()














def main():

    App()



if __name__ == '__main__':
    main()  # Call main function

