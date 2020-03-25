# Squat-Detection
This is to detect if a person in a video image frame squats or not.

# Step 1 includes initializing and loading Openpose on colab
# Step 2 includes generating keypoints from pose detected videos saved in avi and then mp4 formats
# step 3 includes generating image frames in .JPEG format for each pose detected video files
# step 4 inludes creating train and test data sets from image frames 
#..190 training images are used. 100 for no-squat and 90 for squat
# step 5 includes creating test image data sets from image frames
#..16 test images are randomly used. 7 for no-squat and 9 for squats 
# step 6 inludes training a modiefied VGG16 network 
# step 7 inludes testing the modified VGG16 network on 16 image frames
# step 8 inludes network evaluation based on confusion metrics and f-score
# The final output results in two files. One is JSON file (trained_data_squats.json) and the other is header file (trained_data_squats.h5)
#..JSON file is the trained model and header file contains weights of trained model
# Trained model can be loaded along with weigths and then evalueted based on test data image frames from test video data
