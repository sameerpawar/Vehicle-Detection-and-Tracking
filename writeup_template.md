## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car_example.png
[image2]: ./examples/HOG_features_YCrCb_car_not_car.png
[image4]: ./examples/video_frame_car_detected.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./test_videos_output/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the cell-2 of the IPython notebook VehicleDetection.ipynb.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and didnt see much gain in increasing the HOG feature vector dimension so chose the default settings from lectures of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC(). The corresponding code can be found in cell-3 of the IPython notebook VehicleDetection.ipynb.  

I used LinearSVC() with YCrCb color space, HOG features for all channels with parameters `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, color histogram as well as spatial binning feature. 

When tested the stand alone SVM classifier on test data, I found following performance
1. YCrCb
    - test performance = 0.9840
1. LUV
    - test performance = 0.9893
1. HSV
    - test performance = 0.9907
1. YUV
    - test performance = 0.9870
1. HLS
    - test performance = 0.9831
1. RGB
    - test performance = 0.9780

While the stand alone performance of all the color spaces is not much different, I found that when test on video frames YCrCb performed best while trading off between false alarm and missed detection.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search is implemented in cell-1 in the function find_cars(). 

I used 3 scale values [1, 1.5, 4]. I also used a region of interest in form of 

        xstart = 250
        xstop = 1280
        ystart = 400
        ystop_array  = [550, 550, 650]
where ystop values are chosen based on scale.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image where the two cars are detected.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_videos_output/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I implemented a class called BboxHistory() in cell-1 of the code. In this class I keep track of two lists. First is the list of bounding boxes detected positively by classifier as cars and second is the list of cars detected so far. I use a threshold on the first list to eliminate the false alarms. Sometimes the threshold elimination on first list can result in loss of previously detected car. So I use the second list to keep track of the detected cars position and apply a lower threshold to heatmap after masking with only car locations detected in earlier frames. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Essentially in this project I rely on setting correct thresholds to eliminate false alarms without compromising the missed detection rate. I also use averaging over past frames for both eliminating false alarm as well as cleaning heatmap for previously detected cars. This approach does not seem very friendly for generalization to various real world scenarios. Especially I am concerned about any technique that relies on thresholding. I think future direction to pursue would be to see what deep learning based approach gives.  

