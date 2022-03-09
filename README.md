# Summary
* Estimated the fine-grained geolocation from which an image is taken by using computer vision techniques (ViT for feature extraction, LoFTR for points matching, RANSAC for essential matrix computation)
* Ranked 11/215 in the Kaggle competition


# COMP90083_Fine-grained_Localisation

In this project, our task is to recognise the fine-grained geolocation from which an image is taken. We first use a pre-trained vision transform model to extract the global scene features of the input image. Then, those extracted features are used to compute the similarity scores between the test images and training images (the images with known position data), and find the top-10 similar training-test image pairs. After that, we use LoFTR, a transformer-based detector-free model, to match points in the image pair. Based on the detected matching points, we use RANSAC to compute the essential matrix of the image pair.  Finally, the essential matrix is used to fine-tune the geolocation coordinates of the test image. Our method predicts very precise geolocation coordinates and ranked 11/215 in the Kaggle competition.

**Kaggle competition link:** https://www.kaggle.com/t/18c668eea8784f89b8d1af703856f55a

# Authors

Chen-An Fan
Email: chenanf@student.unimelb.edu.au

Hailey Kim
Email: heewonk@student.unimelb.edu.au

# Training 

In training, we divided the provided train set (7500 images) into a development set (6750 images) and a validation set (750 imges).



# Testing

We use the whole test set (1200 images) for testing.



# Test Set Prediction Result Files

These are the prediction result on the test set for the 5 approach implemented in the report.

- test result-method1-top1-ViT-similar.csv
- test result-method2-most-loftr-pts.csv
- test result-method3-affine_regression.csv
- test result-method4-essential-matrix.csv
  - This is our best model, we used this file as the final submission to Kaggle
- test result-method5-camera-pose.csv

You can find these result in the zip file or on the [google drive](https://drive.google.com/drive/folders/1HFDP_HnPZg4OgDZ1k5pFSZb2JsdQW_l-?usp=sharing).

# Code

## How to Run

The code are in the format of jupyter notebook. Just click run all cell, the notebook is ready to run. For more information, please follow the instruction inside the notebook.

The train and test images can be found at https://drive.google.com/drive/folders/1RgMBLT8QGwgrI4d1e7xaNuQiWKAxYHZg?usp=sharing under the folder "train_img" and "test_img"



## final_model_camera_transformation.ipynb

This code contain the approach 4 and 5 written in the report. Those approaches use the camera transformation as the geometric constraints to further improve the coordinates prediction.

Approach 4 (essential matrix) perform the best and we will use it as our final model.

### File Structure

#### Necessary Files

For running the final_model_camera_transformation.ipynb code, you need to have the following file structure. Those file can be accessed from https://drive.google.com/drive/folders/1RgMBLT8QGwgrI4d1e7xaNuQiWKAxYHZg?usp=sharing



- CV_final_project/

  - train.csv

    > This file stores the train image ID and ground truth x,y

  - imagenames.csv 

    > This file stores the test image ID

  - final_model_camera_transformation.ipynb

  - train_img/

    - ... (all training image here)

  - test_img/

    - ... (all test image here)

  - test_sim_dic.json 

    > This file stores the similarity scores between every train image and test images.

  - train_sim_dic.json 

    > This file stores the similarity scores between every pair of train images.



#### Output Files

Those files are the result of running the code "final_model_camera_transformation.ipynb"

- CV_final_project/

  - pred_result_valid.json 

    > This file sotred the prediction result (x,y,error) of the valid set

  - pred_result_test.json 

    > This file sotored the prediction result (x,y) of the test set

  - bad_target_img_valid.json 

    > This file stored the image name of the validation images that all of its top 10 similar images in the development set has matching points fewer than 6 with this image. (Can not find essential matrix)

  - large_error_img_valid.json 

    > This file stored the image name of the validation images that the predicted x,y has MAE larger than 10

  - bad_target_img_test.json 

    > This file stored the image name of the test images that all of its top 10 similar images in the train set has matching points fewer than 6 with this image. (can not find essential matrix)

  - test_result_E.csv 

    > This file is the predicted result (x,y) of all the test images by using the essential matrix approach

  - test_result_recover.csv 

    > This file is the predicted result (x,y) of all the test iamges by using the camera pose recover approach



## affine+regression.ipynb

This code contain the approach 3 of the report. This approach computes the affine matrix between image pairs and then feeds the affine matrix with the x,y coordinates of the train image to a linear regression model to predict the coordinates of the test image.

To shorten the run time, you may load the pre-computed "loft_train_dic.json", "loft_test_dic.json", "train_sim_dic.json" and "test_sim_dic.json" result. Those file can be accessed from https://drive.google.com/drive/folders/1RgMBLT8QGwgrI4d1e7xaNuQiWKAxYHZg?usp=sharing

