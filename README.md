Squat-Detection:-
This project helps in classifying a video image frame in to "squat" or "no squat". It uses VGG16 architecture as a feature extractor
and LSTM architecture as a binary classifier. All the codes were implemented on Google-Colaboratory. 
The first step inludes generation of image frames from training video data set.Use the file "image_frames_generation.py" to store 
video image frames in .jpeg format. All these are stored in a folder "frames". For this use the command "!mkdir frames" and use the 
appropriate folder path in the above python file.
The next step is to zip the folder and download it. For this use the file "frames_zip.txt". Once the image frames are downloaded then
randomly choose 200 images for training and 16 images for testing or validation. Label training and test data sets in ascending order
and arrange such that first 100 belongs to "no squat" and remaining 100 belongs to "squat". Similarly, for the validation/test data
set, arrange such that first 7 images blong to label "no squat" and remaining 9 belongs to "squat".Then use "!mkdir train", "!mkdir test"
commands to create train and test folders. After that upload training data sets to "train" folder and test data sets to "test" folder.
Now we are ready to train and validate the model. Use the file "squat_detect_vgg16_with_lstm.py" for this. The output of this program generates "validation accuracy", "confusion matrix" and "classification report" with f-1 score.It also generates two graphs: training loss v/s validation loss and training accuracy v/s validation accuracy along the "epochs" axis.The program also saves the trained model as "trained_data_squats.json" and trained weights are saves as "trained_data_squats.h5".
Finally, these trained model nad trained weight files are zipped and downloaded using the code in file "header_and_json_files.txt".
Eventually, one can simply use these trained model and weights to test on his/her custom video data set. For this one can simply use the file "model_evaluate.py". However, in order to use this, one has to first create video frames from his/her video data following the above procedure for generating video frames and then create a new folder "newtest' using "!mkdir newtest" and again create randomly chosen 16 images for model evaluation where first 7 frames belong to "no squat" and remaining 9 frames belong to "squat". 

Please use "squat_detect_latest_keypoints_to_LSTM_5_9_2020_Final.ipynb" to detect squat video frames using OpenPose Keypoints (25 keypoints) with an LSTM network.

