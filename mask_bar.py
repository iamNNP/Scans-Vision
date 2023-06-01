import cv2
import numpy as np

def run_mask_bar(image):
    
    def passing(arg):
        pass

    cv2.namedWindow('image')

    cv2.createTrackbar('HMin', 'image', 0, 179, passing)
    cv2.createTrackbar('SMin', 'image', 0, 255, passing)
    cv2.createTrackbar('VMin', 'image', 0, 255, passing)
    cv2.createTrackbar('HMax', 'image', 0, 179, passing)
    cv2.createTrackbar('SMax', 'image', 0, 255, passing)
    cv2.createTrackbar('VMax', 'image', 0, 255, passing)

    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):

        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        result = cv2.inRange(hsv, lower, upper)
        # result = cv2.bitwise_and(image, image, mask=mask)

        if (phMin != hMin) or (psMin != sMin) or (pvMin != vMin) or (phMax != hMax) or (psMax != sMax) or (pvMax != vMax):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            return hMin, sMin, vMin, hMax, sMax, vMax

cv2.destroyAllWindows()