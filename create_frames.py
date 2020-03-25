#Generating image frames for each video files
import cv2
vidcap = cv2.VideoCapture('output_sq4.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("/content/frames/output_sq4_frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  count += 1
