# P4-CarND

### This document has been prepared as per the template provided for the writeup. It considers the [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points and all the material is arranged accordingly.
---
**Advanced Lane Finding Project**

The goals/steps of this project are the following:

* Summaraize the whole implementation (as enumerated below) in a write-up
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Calibration"
[image2]: ./examples/01_OriginalImages.png "Original"
[image3]: ./examples/02_UndistortedImages.png "Undistorted Images"
[image4]: ./examples/03_ColoredBinaryImages.png "Colored Binary Images"
[image5]: ./examples/04_CombinedBinaryImges.png "Combined Binary Imges"
[image6]: ./examples/05_TopViewImages.png "Top View Images"
[image7]: ./examples/06_PolyfitImages.png "Polyfit Images"
[image8]: ./examples/07_RoadPerspectiveImages.png "Road Perspective Images"
[image9]: ./examples/08_WithRoadMarkersImages.png "With Road Markers Images"
[video1]: ./outpt_videos/project_video.mp4 "Video"

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

#### 1. Provide a Writeup/README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file "MyLaneFinder_SaveFiles.py" located in "./examples/". This file works on calibrating the camera and processing the still frames and saving them for ready reference.  

First the camera calibration is done (in line#35 to line#152). Based on a flag, either a new camera setting be calibrated or the camera matrix and distortion coefficient of a pre-calibrated camera be directly used. The camera has been calibrated by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard of size 9x6 is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The corner marked files are stored in "./camera_cal/cornerSavedFiles/" folder.

I then used the output `objpoints` and `imgpoints` to compute the camera matrix and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)
#### 1. Provide examples of distorted images taken from a real camera.
Below are some of the images of snaps taken from a real camera. Therefore, the images do have certain distortions (tangential and radial) inherent to them. They are so subtle that hardly they will be perceivable unless the un-distortion is carried out. 

![alt text][image2]
#### 2. Provide an example of a distortion-corrected image.

To demonstrate the un-distortion of images, the "cv2.undistort(img,mtx,dist,None,mtx)" function from opencv library has been used. This takes the camera matrix and the distortion coefficients as input for undistorting the images. This is done on line#552 in the code "MyLaneFinder.py" in the "./examples/" folder. The undistorted images corresponding to the above mentioned original images are shown below:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

This is done on line#554-556 with the function "sobelFilteredImg(undist)" in the code "MyLaneFinder.py" in the "./examples/" folder. It takes the undistorted image from the previous step and generate a binary image. s_channel threshholding on the hsv-space and sobel filter based gradient filtering of the l-channel is used for generating this binary image. The binary images corresponding to the above mentioned original images are shown below:

![alt text][image5]
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is acheived with the function `getTopView(combined_binary)`, which appears in lines#558 of the code.  This function takes as inputs the binary image from the previous step, as well as source (`src`) and destination (`dst`) points to produce the view of the road as if viewed from the top.  I chose the hardcoded source and destination points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 541,489      | 290,324        | 
| 746,489      | 990,324      |
| 777,511     | 990,396      |
| 511,511      | 290,396        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Once the Perspective transform matrix M and the Inverse Perspective transform matrix Minv are evaluated, they have been frozen and are used for all the further processing af all the frames. The top view images corresponding to the above mentioned original images are shown below:

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is done on line#560 with the function "fit_polynomial(topViewImg)" in the code "MyLaneFinder.py" in the "./examples/" folder. This function takes the Top View Image as input and extracts the lane-line pixels using two kinds of methods. When an existing fitted polynomial is available, the search is carried out about this curve. Otherwise the window based search from scratch is performed on a frame. The detected lane-line pixels along with the fitted polynomial images corresponding to the original images are shown below:

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is done on line#560 with the function "getFinalMarkedRoad()" in the code "MyLaneFinder.py" in the "./examples/" folder. This function takes the PolyFit image of the top view of the road and the inverse Perspective Transform matrix as input and calculates the radius of curvature of the left and the right lane separately. It also calculates the offset of the vehicle form the center of the lane. It calculates the road map in the perspective view of the camera as well. The projected perspective view of the road markings corresponding to the original images are shown below:

![alt text][image8]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally the perspective projection of the road marking is superimposed on the undistorted image of the first step. The undistorted images with superimposed projected perspective road markings corresponding to the original images are shown below:


![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Using the same pipeline as described above, video files are processed frame by frame after extracting frames using the functions "VideoFileClip(fname)" and "fl_image(roadMarkingPipeline)". Then the processed image files are stiched together to make the final video using the "write_videofile() function". The output of the above processing is available at the bolow link:

Output Video: [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major issues that I am facing are as below:
1. The pipeline is taking more time than the time between frames for a video of even 25 fps. This a serious drawback as in this form it cannot be used in realtime application.
2. The pipeline fails to capture on the challenge and the harder challenge videos. The reason for the first case is its failure to distingush road color from that of the lane lines. In th esecond case, the high value of curvature is creating problem. Even when I tries the convolution method the lane-line pixel search, it didn't work. I think the solution lies in increased width of search windows with reduced height. I shall try that. However, that will still increase the time taken to process each frame. 
