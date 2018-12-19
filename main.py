import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
import matplotlib.image as mpimg
from lane_finding import lane_finding_pipeline

if __name__ == '__main__':

    verbose = True
    #if verbose:
       #plt.ion()
       # figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()

    input_image_list = os.listdir("test_images/")
    input_image_dir = "./test_images/"
    input_image_list = [join(input_image_dir, name) for name in input_image_list]
    image_out_dir = 'test_images_output'


    for image in input_image_list:

        img_out = lane_finding_pipeline(image)
        out_path = join(image_out_dir, basename(image))
        mpimg.imsave(out_path, img_out)

        print('Currently processing image: {}'.format(image))