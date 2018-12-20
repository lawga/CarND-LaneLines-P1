import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    blank_image = np.copy(img)*0 # creating a blank to draw lines on


    x1AvgL = 0
    x2AvgL = 0
    y1AvgL = 0
    y2AvgL = 0
    x1AvgR = 0
    x2AvgR = 0
    y1AvgR = 0
    y2AvgR = 0
    

    startYL = np.float32(320.0)
    endYL = np.float32(540.0)
    startXL = np.float32(485.0)
    endXL = np.float32(857.0)

    startYR = np.float32(320.0)
    endYR = np.float32(540.0)
    startXR = np.float32(472.0)
    endXR = np.float32(162.0)

    nL = 1
    nR = 1
    gh = 0
    for line in lines:
<<<<<<< HEAD
        #print(line)
        for x1,y1,x2,y2 in line:
            if (y1!=y2):
                if ((x2-x1)/(y2-y1)) > 1 and ((x2-x1)/(y2-y1)) < 3 :
                    x1AvgL = x1AvgL + (x1 - x1AvgL)/nL
                    x2AvgL = x2AvgL + (x2 - x2AvgL)/nL
                    y1AvgL = y1AvgL + (y1 - y1AvgL)/nL
                    y2AvgL = y2AvgL + (y2 - y2AvgL)/nL
                    #print(x1AvgL, x2AvgL, y1AvgL, y2AvgL)
                    #print((x2-x1)/(y2-y1))
                    nL += 1
                elif ((x2-x1)/(y2-y1)) < -1 and ((x2-x1)/(y2-y1)) > -3  :
                    x1AvgR = x1AvgR + (x1 - x1AvgR)/nR
                    x2AvgR = x2AvgR + (x2 - x2AvgR)/nR
                    y1AvgR = y1AvgR + (y1 - y1AvgR)/nR
                    y2AvgR = y2AvgR + (y2 - y2AvgR)/nR
                    #print((x2-x1)/(y2-y1))
                    nR += 1
  
    if (y1AvgL== y2AvgL) or (y1AvgR==y2AvgR):
=======

         for x1,y1,x2,y2 in line:
            gh +=1
            if (x2-x1)/(y2-y1) > 0 :
                x1AvgL = x1AvgL + (x1 - x1AvgL)/nL
                x2AvgL = x2AvgL + (x2 - x2AvgL)/nL
                y1AvgL = y1AvgL + (y1 - y1AvgL)/nL
                y2AvgL = y2AvgL + (y2 - y2AvgL)/nL
                nL += 1
            elif (x2-x1)/(y2-y1) < 0 :
                x1AvgR = x1AvgR + (x1 - x1AvgR)/nR
                x2AvgR = x2AvgR + (x2 - x2AvgR)/nR
                y1AvgR = y1AvgR + (y1 - y1AvgR)/nR
                y2AvgR = y2AvgR + (y2 - y2AvgR)/nR
                nR += 1
  
    if (x1AvgL==0 and x2AvgL==0 and y1AvgL==0 and y2AvgL==0) or (x1AvgR==0 and x2AvgR==0 and y1AvgR==0 and y2AvgR):
>>>>>>> 6ddd6b3f34b6808ca5dbac8393ee391af2a562c9
        print(x1AvgL, x2AvgL, y1AvgL, y2AvgL)
        print(x1AvgR, x2AvgR, y1AvgR, y2AvgR)
    else:
        [slopeL, interceptL] = np.polyfit([x1AvgL, x2AvgL], [y1AvgL, y2AvgL], 1)
        [slopeR, interceptR] = np.polyfit([x1AvgR, x2AvgR], [y1AvgR, y2AvgR], 1)

        #print(interceptL, slopeL)
        #print(interceptR, slopeR)

        startYL = np.float32(320.0)
        endYL = np.float32(540.0)
        startXL = np.float32((startYL - interceptL) / slopeL)
        endXL = np.float32((endYL - interceptL) / slopeL)

        startYR = np.float32(320.0)
        endYR = np.float32(540.0)
        startXR = np.float32((startYR - interceptR) / slopeR)
        endXR = np.float32((endYR - interceptR) / slopeR)

        #print(startXL, endXL, startXR, endXR)
        ####################################################################3
    cv2.line(blank_image, (startXL, startYL), (endXL, endYL), color, thickness, lineType=4, shift=0)
    cv2.line(blank_image, (startXR, startYR), (endXR, endYR), color, thickness, lineType=4, shift=0)
    
    
    return blank_image


<<<<<<< HEAD
=======
    #print(startXL, endXL, startXR, endXR)
    ####################################################################3
    cv2.line(img, (startXL, startYL), (endXL, endYL), color, thickness, lineType=4, shift=0)
    cv2.line(img, (startXR, startYR), (endXR, endYR), color, thickness, lineType=4, shift=0)
    return img


>>>>>>> 6ddd6b3f34b6808ca5dbac8393ee391af2a562c9

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def lane_finding_pipeline(img, vertices):
    
    # Read in the image
    #img = mpimg.imread(img)
    

    #grayscale the image
    gray = grayscale(img)
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 40
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    #plt.imshow(edges)
    
    #plt.axis('off')

    #plt.show()

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = img.shape
    #print (imshape)
    #vertices = np.array([[(180,imshape[0]-50),(480, 320), (530,320), (825,imshape[0]-50)]], dtype=np.int32)
    #vertices = np.array([[(100,imshape[0]),(450, 320), (520,320), (925,imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    polygin = cv2.polylines(img, [vertices], True, (0,255,255),4)
    
    #plt.imshow(polygin)
    
    #plt.axis('on')

    #plt.show()

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 10   # maximum gap in pixels between connectable line segments
    line_image = np.copy(img)*0 # creating a blank to draw lines on
    Hough_lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    stright_lines = draw_lines(img, Hough_lines, color=[255, 69, 0], thickness=8)
    transparent = weighted_img(img, stright_lines, α=0.8, β=1., γ=0.)
    
<<<<<<< HEAD
    return transparent
=======
    img_out = draw_lines(img, lines, color=[255, 69, 0], thickness=8)
    
    return img_out
>>>>>>> 6ddd6b3f34b6808ca5dbac8393ee391af2a562c9
