[//]: # (Image References)

[image1]: ./report_images_output/gray.jpg "Grayscale"
[image2]: ./report_images_output/blur_gray.jpg "Blured Grayscale"
[image3]: ./report_images_output/edges.jpg "Edges"
[image4]: ./report_images_output/masked_edges.jpg "Masked Edges"
[image5]: ./report_images_output/line_image.jpg "Lines"
[image6]: ./report_images_output/lines_edges_array.jpg "Filtered Lines"
[image7]: ./report_images_output/output.jpg "Output"

[image8]: ./challenge_images_output/challenge_Moment.jpg "Output Challange"



# **Finding Lane Lines on the Road** 

---

**"Final - Fixed all.ipynb" is the final version for the project.**

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on my work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. First, I converted the images to grayscale:
![Grayscale][image1]

then I blur the image to filter out any smll areas where the clolours transition  in a big values:
![Blured Grayscale][image2]

After that I detect all the edges in the image:
![Edges][image3]

Then I filter the area of inerest to work and detect the lanes later on:
![Masked Edges][image4]

I Draw the detected lines on an empty enviroment to work on:
![Lines][image5]


I projected the detected lines on the orginal image to get a better understanding of what each detected line represent:
![Filtered Lines][image6]


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calculating the slopes of the lines and filter the ones that satisfy the slope that represent a left/right lanes then group th e lines with negative valuse and avrage them and do the same for the lines with positive values.

After finding the avrage slope, I draw two big lines that represent the detected lanes on the original image.

![Output][image7]



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be that the lines drwan on the vedio are jerky when going from one frme to another. 

Another shortcoming could be that the method used is a liitle bit slow and may also not detect all the lanes on the street which cause the drwan line to be a little bit tilted/shifted away from the actual lanes.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be that have a better parameters for the hough lines filter. whcih may help in filtering out noise and detect more lanes.

Another potential improvement could be that the line fitting method used is the not the most efficent way. but its the one I  understood. Maybe if I spent more time on that area I would've come with a better solution.

### 4. Challenge

I think I did a decent work with tuning my pipline to be able to detect lanes in the challenge vedio. another trick was resizing the vedio stream to a smaller size. and also the polygon edges also needed to be changed to focus on the area of interest in the vedio, and not forgetting retunning the hough lines detector:

![Challenge][image8]


