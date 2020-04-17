# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:27:37 2020

@author: pappuyadav
"""

#Generating image frames for each video files in avi format. 
#Upload your test video file (in avi format) in the home folder then replace 'sq8.avi' with your video file
import cv2
vidcap = cv2.VideoCapture('/content/sq8.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("/content/frames/sq8_frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1