import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
import matplotlib.image as mpimg
from lane_finding import lane_finding_pipeline

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML