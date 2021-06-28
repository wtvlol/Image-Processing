import cv2 as cv
import numpy as np
import PIL
import matplotlib.pyplot as plt

#image is the file path
#threshold is a list of threshold values, from lowest to greatest, 
#range from 0 to 1 with reference to the mean value of the pixels and set True as default
#coords specificies region which threshold is applied, cropping tool of sorts, consisting of 2 tuples
#mean is used to specify whether the mean values of the grayscale image is used or 8 bit colour, 
#if mean is True and threshold of value 1, it will default the threshold to the average value
def thresholdSegment(image,threshold,coords = [],mean = True):
    
    pre = cv.imread(image)
    #image cropping
    if coords != []:
        x_min = coords[0][1]
        x_max = coords[0][1]+coords[1][1]
        y_min = coords[0][0]
        y_max = coords[0][0]+coords[1][0]
        im = pre[x_min:x_max,y_min:y_max]
    else:
        im = pre
        
    #converts image to grayscale
    im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    
    #different grayscale colours
    threshold_cols = [int(255/(len(threshold))+1)*i for i in range(len(threshold)+1)]

    #mean check
    im_shape = im.shape
    im_r = im.reshape(im_shape[0]*im_shape[1])
    
    threshold.sort()
    avr = im.mean()
    if mean == True:
        threshold = np.array(threshold) * avr 
    elif mean == False:
        threshold = np.array(threshold) * 255
        
        
    #thresholding
    for i in range(im_r.shape[0]):
        for k in range(len(threshold))[::-1]:
            if im_r[i] >= threshold[k]:
                im_r[i] = threshold_cols[k]
                break
            elif im_r[i] < threshold[k]:
                im_r[i] = threshold_cols[k-1]
                break

    """
    for i in range(len(im)):
        for t in range(len(im[0])):
            for k in range(len(threshold))[::-1]:
                if im[i][t] >= threshold[k]:
                    im[i][t] = threshold_cols[k]
                    break
                if im[i][t] < threshold[k]:
                    im[i][t] = threshold_cols[k-1]
                    break
    """
    
    im = im_r.reshape(im_shape[0],im_shape[1])
    print([set(i) for i in im])
    plt.imshow(im)
    ###plot historgram
#thresholdSegment("testImages/1.jpeg",[0.25,0.5],mean = True)

def edgeContours(image):
    pre = cv.imread(image)
    im = cv.imread(image)
    
    #converts image to grayscale
    im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    
    #uses average value as global threshold
    _,im = cv.threshold(im,im.mean(),255,cv.THRESH_BINARY)
    
    #finds the contours
    contours,hierachy = cv.findContours(im,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    print(hierachy)
    ###object snapshot?
    ###find highest order parent contour
    
    #convert back to RGB to draw the contours in colour
    im = cv.cvtColor(im,cv.COLOR_GRAY2RGB)
    
    #drawing the contours
    cv.drawContours(im,contours,-1,(0,255,0),2)
    
    #display
    plt.imshow(im)
    
    """
    cv.imshow('contours', im)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
    """

edgeContours("testImages/3.png")


#colour image segmentation
