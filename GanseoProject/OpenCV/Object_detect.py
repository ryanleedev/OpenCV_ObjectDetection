
import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture("1_1.mp4")
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print(frames_count, fps, width, height)

history = 500
varThreshold = 40
detectShadow=False
sub = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadow)  #MOG2 algorithm for background removal

ret, frame = cap.read()  #import as image
ratio = 0.8  #Ratio setting
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  #Sizing the image to the set ratio
width2, height2, channels = image.shape



while True:
    ret, frame = cap.read() 
    if ret: 
        image = cv2.resize(frame, (0, 0), None, ratio, ratio) 
        cv2.imshow("image", image)
        roi = image[:,250:500]  #area of interest

        grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  #Convert image to black and white
        cv2.imshow("gray", grayROI)
        gmask = sub.apply(grayROI)
        cv2.imshow("gmask", gmask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  #Apply morphology transformation
        closing = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, kernel)  #Closing technique
        cv2.imshow("closing", closing) 
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel) #Opening technique 
        cv2.imshow("opening", opening) 
        dilation = cv2.dilate(opening, kernel)   #1st round of dilation
        cv2.imshow("dilation", dilation)   
        dilation2 = cv2.dilate(dilation, kernel)    #2nd round of dilation
        cv2.imshow("dilation2", dilation2) 
        _, final = cv2.threshold(dilation2, 220, 255, cv2.THRESH_BINARY) #remove shadows

        cv2.line(image, (300,150), (450,150), (0,0,255), 2) #Caution line (red)
        # creates contours
        # cv2.imshow('bins',bins)
        contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        #Contour Recognition Minimum Range
        minarea = 400
        #Contour Recognition Maximum Range
        maxarea = 50000
                
        for i in range(len(contours)):  # Show all outlines of a frame
                area = cv2.contourArea(contours[i])  # Contour range
                if minarea < area < maxarea:  # Contour Max Min Range
                    # Calculate contour center point
                    cnt = contours[i]

                    #Calculate the image moments through the cv2.moments function and return them in the form of a dictionary. 
                    # The returned moment is a total of 24, consisting of 10 positional moments, 7 central moments, and 7 normalized 
                    # central moments.
                    # - (Spatial Moments) : m00, m10, m01, m20, m11, m02, m30, m21, m12, m03
                    # - Central Moments : mu20, mu11, mu02, mu30, mu21, mu12, mu03
                    # - Central Normalized Moments : nu20, nu11, nu02, nu30, nu21, nu12, nu03
                    M = cv2.moments(cnt)
                    #The formula to find the center point of contours
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Create a rectangle outline
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    #coordinates of the lower center of the outline
                    xMid = int((x+ (x+w))/2) 

                    #Yellow dot at the center of the outline
                    cv2.circle(roi, (xMid, y+h), 5,(0,255,255))

                    #Yellow dot at the lower center of the outline A warning sign is displayed when the range is exceeded
                    if (y+h > 150):
                        cv2.putText(roi, str("WARNING"),(cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("countours", image)
    key = cv2.waitKey(20)
    if key == 27:
       break

cap.release()
cv2.destroyAllWindows()