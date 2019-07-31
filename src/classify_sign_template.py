#Qudus Agbalaya
#references:
#https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
#https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#OpenCV Version: 3.0

import cv2
import numpy as np

WARPED_XSIZE = 200
WARPED_YSIZE = 300

#set threshold and threshold for canny
canny_thresh = 120
threshold = 0.5


VERY_LARGE_VALUE = 100000

NO_MATCH            =  0
STOP_SIGN           =  1
SPEED_LIMIT_40_SIGN =  2
SPEED_LIMIT_80_SIGN =  3
SPEED_LIMIT_100_SIGN = 4
YIELD_SIGN           = 5 

def show_image_simple(image):
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_on_image(image, text):
    fontFace = cv2.FONT_HERSHEY_DUPLEX;
    fontScale = 2.0
    thickness = 3
    textOrg = (10, 130)
    cv2.putText(image, text, textOrg, fontFace, fontScale, thickness, 8);
    return image

#funtion to resize image
#from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
#used this because I was getting errors when matchng templates because they weren't the same dimensions

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

#order_points to get and order cordinates
#from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    rect = np.zeros((4,2),dtype ="float32")

    s = pts.sum(axis =1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

#function to apply perspective transform
#from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl,tr,br,bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0,0],
        [maxWidth -1, 0],
        [maxWidth -1, maxHeight -1],
        [0,maxHeight -1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
    
def identify():
    #image = cv2.imread("speedsign3.jpg")
    #image = cv2.imread("speedsign4.jpg")
    #image = cv2.imread("speedsign14.jpg")
    image = cv2.imread("speedsign16.jpg")
    #image = cv2.imread("yield_sign1.jpg")
    #image = cv2.imread("stop4.jpg")
    forty_template = cv2.imread("speed_40.bmp")
    print('forty template',forty_template.shape)
    forty_template = cv2.cvtColor(forty_template, cv2.COLOR_BGR2GRAY )
    forty_template = image_resize(forty_template, height = 200)
    
    eighty_template = cv2.imread("speed_80.bmp")
    print('eighty template',eighty_template.shape)
    eighty_template = cv2.cvtColor(eighty_template, cv2.COLOR_BGR2GRAY )
    eighty_template = image_resize(eighty_template, height = 200)
    
    one_hundred_template = cv2.imread("speed_100.bmp")
    print('one hundred template',one_hundred_template.shape)
    one_hundred_template = cv2.cvtColor(one_hundred_template, cv2.COLOR_BGR2GRAY )
    one_hundred_template = image_resize(one_hundred_template, height = 200)

    print("Reading forty template")
    show_image_simple(eighty_template)
    print("Reading eighty template")
    show_image_simple(forty_template)
    print("Reading one hundred template")
    show_image_simple(one_hundred_template)

    image_original = image.copy()
    image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    image_bw = image.copy()
    image = cv2.blur( image, (3,3) )
    sign_recog_result = NO_MATCH

    # here you add the code to do the recognition, and set the variable
    # sign_recog_result to one of STOP_SIGN, SPEED_LIMIT_40_SIGN, SPEED_LIMIT_80_SIGN, or NO_MATCH
    #find canny edges
    canny_thresh = 120
    edgeimg = cv2.Canny(image, canny_thresh, canny_thresh*2, apertureSize=3)

    #find contours
    _, contours, _ = cv2.findContours(edgeimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    big_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    #determine type of sign

    for c in big_contours:
        img_contour = image_original.copy()
        cv2.drawContours(img_contour, [c], 0, (0,255,0), 3)
		
        #used 0.008 for all signs except yield
        peri = 0.008 * cv2.arcLength(c,True)
        c = cv2.approxPolyDP(c, peri, True)

        if len(c) == 3:
            print 'yield sign'
            sign_recog_result = 5
            break
        elif len(c) == 8:
            print 'stop sign'
            sign_recog_result = 1
            break
        elif len(c) == 4:
            w = four_point_transform(image,c.reshape(4,2))
            w = image_resize(w,height =200)
            
            #match template for forty and get min and max values
            result_40 = cv2.matchTemplate(forty_template, w, cv2.TM_CCOEFF_NORMED)
            min_val_40, max_val_40, min_loc_40, max_loc_40 = cv2.minMaxLoc(result_40)

            #match template for eighty and save min and max values
            result_80 = cv2.matchTemplate(eighty_template, w, cv2.TM_CCOEFF_NORMED)
            min_val_80, max_val_80, min_loc_80, max_loc_80 = cv2.minMaxLoc(result_80)

            #match template for hundred and get min and max values
            result_100 = cv2.matchTemplate(one_hundred_template, w, cv2.TM_CCOEFF_NORMED)
            min_val_100, max_val_100, min_loc_100, max_loc_100 = cv2.minMaxLoc(result_100)

            if(max_val_40 > threshold):
                print("Stop sign 40 detected...")
                sign_recog_result = 2
                break
            elif(max_val_80 > threshold):
                print("80 sign detected")
                sign_recog_result = 3
                break
            elif(max_val_100 > threshold):
                print("100 sign detected")
                sign_recog_result = 4
                break



    show_image_simple(image_original)
    show_image_simple(image)


    if sign_recog_result == NO_MATCH:
        sign_string = "No match"
        file_string = "No_match.jpg"
    elif sign_recog_result == STOP_SIGN:
        sign_string = "Stop sign"
        file_string = "Stop_sign.jpg"
    elif sign_recog_result == SPEED_LIMIT_40_SIGN:
        sign_string = "40_sign"
        file_string = "40_sign.jpg"
    elif sign_recog_result == SPEED_LIMIT_80_SIGN:
        sign_string = "80_sign"
        file_string = "80_sign.jpg"
    elif sign_recog_result == SPEED_LIMIT_100_SIGN:
        sign_string = "100_sign"
        file_string = "100_sign.jpg"
    elif sign_recog_result == YIELD_SIGN:
        sign_string = "Yield sign"
        file_string = "Yield_sign.jpg"

    # save the results
    print(sign_string)
    result = write_on_image(image_original, sign_string)
    cv2.imwrite(file_string, result)


identify()
