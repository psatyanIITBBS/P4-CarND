import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from moviepy.editor import VideoFileClip
import os     
import time

# Three types of camera calibration for three types of videos taken from different cameras. The respective Camera 
# matrix and distortion coefficient will be used. Here the data from Udacity is being used.
VF_IR_Bullet_8mm = False
Udacity = True
GoPro = False

# The following flags are for processing the still frames only. That is why all the flags now are set to false.
# For using a pre-calibrated camera, set this flag to False. For first time calibration, set it to True.
CameraCalibrateFlag = False
# For using an already undistorted images, set this flag to False. For undistorting for the first time, set it to True.
undistortFlag = False
# For using images that have already been converted to color-binary, set this flag to False. 
# For creating the color-binaries for the first time, set it to True.
colorBinaryFlag = False
# For using images that have already been converted to top view, set this flag to False. 
# For creating the top view for the first time, set it to True.
getTopViewFlag = False
# For using images that have already been converted to Plyfit view, set this flag to False. 
# For creating the Plyfit view for the first time, set it to True.
getPolyfitFlag = False
# For using images that have already been converted to RoadWithMarker view, set this flag to False. 
# For creating the RoadWithMarker view for the first time, set it to True.
getWithRoadMarkersFlag = False
# To save the above views for th efirst time in their respective folders, set this flag to True.
saveFlag = False
#..................................................................................
# For using a new camera for which calibration data is not available
# Run the calibration job only once for a given camera and store the 
# camera matrix and the distortion corfficients for the camera for later use.
if CameraCalibrateFlag == True:
    if VF_IR_Bullet_8mm == True:
        nx = 13
        ny = 7
    if Udacity == True:
        nx = 9
        ny = 6
    if GoPro == True:
        nx = 7
        ny = 7
    # termination criteria for cv2.cornerSubPix() function
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    if VF_IR_Bullet_8mm == True:
        images = glob.glob('../camera_cal/VF_IR_8mmFL_Edited/8mm_*.png')   # VF_IR_Bullet_8mmFL Database
        cornerFilesPath = '../camera_cal/VF_IR_8mmFL_Edited/cornerSavedFiles/'  
        undistFilesPath = '../camera_cal/VF_IR_8mmFL_Edited/undistortSavedFiles/'  
    if Udacity == True:
        images = glob.glob('../camera_cal/calibration*.jpg')   # Udacity Database
        cornerFilesPath = '../camera_cal/cornerSavedFiles/'   
        undistFilesPath = '../camera_cal/undistortSavedFiles/'   
    if GoPro == True:
        images = glob.glob('../camera_cal/GoPro/GOPR13*.jpg')   # GoPro Database
        cornerFilesPath = '../camera_cal/GoPro/cornerSavedFiles/'   
        undistFilesPath = '../camera_cal/GoPro/undistortSavedFiles/'  
    
    # Step through the list and search for chessboard corners
    for fname in images:
        
        # Read the image file in BGR mode
        img = cv2.imread(fname)
        
        # Extract the filename of the image from its path
        filenameOnly = fname.split('\\')[-1]
        
        # Convert to GRAY scale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
    
        # If found, add object points, image point
        if ret == True:
            # Append Object points to the global ObjectPoint list
            objpoints.append(objp)
            
            # Try to get the best point to be marrked as corner
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            # Append Image points to the global ImagePoint list
            imgpoints.append(corners)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imshow('img',img)
            
            # Saves the images marked with corners in a folder
            cornerFilesName = cornerFilesPath + filenameOnly 
            cv2.imwrite(cornerFilesName,img)
            
            # Wait for half a second
            cv2.waitKey(500)
    
    for fname in images:
        img = cv2.imread(fname)
        filenameOnly = fname.split('\\')[-1]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
        undist = cv2.undistort(img,mtx,dist,None,mtx)
        undistFilesName = undistFilesPath + filenameOnly 
        cv2.imwrite(undistFilesName,undist)    
    
    print('Using the a new camera with unknown focal length...')
    print('')    
    print('Camera Matrix = ',mtx)
    print('')
    print('Distortion Coefficiants = ', dist)
#.............................................................................................

#.............................................................................................
# For using pre-calibrated camera 
if CameraCalibrateFlag == False:
    # Camera calibration for VF_IR_Bullet for 8mm Focal Length 
    if VF_IR_Bullet_8mm == True:
        mtx =  [[1.65521864e+03, 0.00000000e+00, 4.78550170e+02], 
                [0.00000000e+00, 1.66185960e+03, 3.35281163e+02], 
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        dist =  [[-2.63844771e-01, -6.01216069e-01, -1.86507107e-03, -3.60339565e-03,2.86392537e+00]]
        print('Using the VF_IR_Bullet camera with 8 mm focal length...')
        print('')
        print('Camera Matrix = ',mtx)
        print('Distortion Coefficiants = ', dist)        
        
    # Camera calibration for Udacity camera (probably 6 mm focal length)
    if Udacity == True:         
        mtx =  np.array([[1.15694047e+03, 0.00000000e+00, 6.65948820e+02],
                [0.00000000e+00, 1.15213880e+03, 3.88784788e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist =  np.array([[-2.37638062e-01, -8.54041477e-02, -7.90999654e-04, -1.15882246e-04,1.05725940e-01]])
        print('Using the Udacity camera with 6 mm focal length...')
        print('')
        print('Camera Matrix = ',mtx)
        print('')
        print('Distortion Coefficiants = ', dist)          
#............................................................................................    

# Define a class to receive the characteristics of left and right detected lane lines 
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for all the detected pixels
        self.allx = None  
        #y values for all the detected pixels
        self.ally = None
        #x values for detected line pixels
        self.fitx = None  
        #y values for detected line pixels
        self.fity = None        

#**********************************************************************************
#The following two functions are helping files for using the convolution based lane pixel detection.	
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    for level in range(1,(int)(image.shape[0]/window_height)):
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        window_centroids.append((l_center,r_center))
    return window_centroids

def find_lane_pixels_conv(warped):
    window_width = 100 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    if len(window_centroids) > 0:
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
        for level in range(0,len(window_centroids)):
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            #asd = 9
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
    ## Extract left and right line pixel positions
    nonzero = l_points.nonzero()
    l.ally = np.array(nonzero[0])
    l.allx = np.array(nonzero[1])     
    nonzero = r_points.nonzero()
    r.ally = np.array(nonzero[0])
    r.allx = np.array(nonzero[1])
    return

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        ## Uncomment these to draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        #(win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),
        #(win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    if len(left_lane_inds) > minpix:
        l.detected = True
    else:
        l.detected = False
    if len(right_lane_inds) > minpix:
        r.detected = True
    else:
        r.detected = False
    # Extract left and right line pixel positions
    l.allx = nonzerox[left_lane_inds]
    l.ally = nonzeroy[left_lane_inds] 
    r.allx = nonzerox[right_lane_inds]
    r.ally = nonzeroy[right_lane_inds]
    return

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 25
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (l.current_fit[0]*(nonzeroy**2) + l.current_fit[1]*nonzeroy + 
                    l.current_fit[2] - margin)) & (nonzerox < (l.current_fit[0]*(nonzeroy**2) + 
                    l.current_fit[1]*nonzeroy + l.current_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (r.current_fit[0]*(nonzeroy**2) + r.current_fit[1]*nonzeroy + 
                    r.current_fit[2] - margin)) & (nonzerox < (r.current_fit[0]*(nonzeroy**2) + 
                    r.current_fit[1]*nonzeroy + r.current_fit[2] + margin)))
    # Set minimum number of pixels found to recenter window
    minpix = 200    
    if len(left_lane_inds) > minpix:
        l.detected = True
    else:
        l.detected = False
            
    if len(right_lane_inds) > minpix:
        r.detected = True
    else:
        r.detected = False                
    # Again, extract left and right line pixel positions
    l.allx = nonzerox[left_lane_inds]
    l.ally = nonzeroy[left_lane_inds] 
    r.allx = nonzerox[right_lane_inds]
    r.ally = nonzeroy[right_lane_inds]
    return

# This function provides the sobel filtered images.
def sobelFilteredImg(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    cut_level = 420
    s_append = np.zeros_like(hls[:,:,1])
    sx_append = np.zeros_like(hls[:,:,1])
    l_channel = hls[cut_level:,:,1]
    s_channel = hls[cut_level:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    s_append[cut_level:,:] = s_binary
    sx_append[cut_level:,:] = sxbinary
    #print(s_append.shape[0])
    return sx_append,s_append

def fit_polynomial(binary_warped):

    # Find our lane pixels first
    if l.detected == True and r.detected == True:
        search_around_poly(binary_warped)
        #find_lane_pixels_conv(binary_warped)        
    else:
        find_lane_pixels(binary_warped)
        #find_lane_pixels_conv(binary_warped)    
    # Populate the recent fit cooefficients if the last frame detected a lane 
    if l.detected == True:
        l.recent_nfit.append(l.current_fit)
        temp = l.recent_nfit.pop(0)
        l.best_fit = np.mean(l.recent_nfit,axis=0)
    if r.detected == True:    
        r.recent_nfit.append(r.current_fit)
        temp = r.recent_nfit.pop(0)
        r.best_fit = np.mean(r.recent_nfit,axis=0)    
    
    l.current_fit = np.polyfit(l.ally, l.allx, 2)
    r.current_fit = np.polyfit(r.ally, r.allx, 2) 
    
    l.diff = ((l.current_fit - l.recent_nfit[-1])/l.recent_nfit[-1])*100
    r.diff = ((r.current_fit - r.recent_nfit[-1])/r.recent_nfit[-1])*100
    
    if np.max(l.diff)>15:
        l.detected = False
    if np.max(r.diff)>15:
        r.detected = False       
    
    # Generate x and y values for plotting
    
    #start_time = time.time()
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        l.fitx = l.current_fit[0]*ploty**2 + l.current_fit[1]*ploty + l.current_fit[2]
        r.fitx = r.current_fit[0]*ploty**2 + r.current_fit[1]*ploty + r.current_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        l.fitx = 1*ploty**2 + 1*ploty
        r.fitx = 1*ploty**2 + 1*ploty

    if l.detected == True:
        l.recent_xfitted.append(l.fitx)
        temp = l.recent_xfitted.pop(0)
        l.bestx = np.mean(l.recent_xfitted,axis=0)
    if r.detected == True:    
        r.recent_xfitted.append(r.fitx)
        temp = r.recent_xfitted.pop(0)
        r.bestx = np.mean(r.recent_xfitted,axis=0)
    return

def getRoad(left_fitx,right_fitx,ploty):
    road = np.zeros((720,1280,3),dtype=np.uint8)
    top_left = [left_fitx[0],0]
    top_right = [right_fitx[0],0]
    bottom_left = [left_fitx[-1],719]
    bottom_right = [right_fitx[-1],719]
    left = ()
    right = ()
    for i in range(len(left_fitx)):
        qwe = [left_fitx[len(left_fitx)-1-i],ploty[len(left_fitx)-1-i]]
        left = left + (qwe,)
        
    for i in range(len(left_fitx)):
        qwe = [right_fitx[i],ploty[i]]
        right = right + (qwe,)
            
    temp = ()
    temp = temp + (top_left,)
    temp = temp + (top_right,)
    temp = temp + (right)
    temp = temp + (bottom_right,)
    temp = temp + (bottom_left,)
    temp = temp + (left)
    
    triangle = np.array([temp], np.int32)
    cv2.fillConvexPoly(road, triangle, [0, 255, 0])
    return road

def getTopView(combined_binary):
    img_size = (combined_binary.shape[1],combined_binary.shape[0])
    img_bot_half = combined_binary[:combined_binary.shape[0]-30,:]
    #FourCorners = [[541,489],[746,489],[777,511],[511,511]]
    #src = np.float32(
        #[FourCorners[0],
        #FourCorners[1],
        #FourCorners[2],
        #FourCorners[3]])
    
    ## Define conversions in x and y from pixels space to meters
    ##ym_per_pix = 30/720 # meters per pixel in y dimension
    ##xm_per_pix = 3.7/700 # meters per pixel in x dimension
    #dst = np.float32([[290,324],[990,324],[990,396],[290,396]])
    #M = cv2.getPerspectiveTransform(src,dst)
    #print('M = ',M)
    #Minv = cv2.getPerspectiveTransform(dst,src)  
    #print('Minv = ',Minv)
    topViewImg = cv2.warpPerspective(img_bot_half,M,img_size,flags=cv2.INTER_LINEAR)
    
    return topViewImg

def getFinalMarkedRoad(ploty,Minv,img):
    global Frame, RadCurveStr
    
    #start_time = time.time()
    ym_per_pix = 30/720 # meters per pixel in y dimension        
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)*ym_per_pix
    
    leftRoadPosition = l.recent_xfitted[-1][-1]
    rightRoadPosition = r.recent_xfitted[-1][-1]
    
    halfLaneWidth = (rightRoadPosition - leftRoadPosition)*xm_per_pix/2
    distFromCenterOfLane = ((640 - leftRoadPosition)*xm_per_pix - halfLaneWidth)
    l.line_base_pos = (640 - leftRoadPosition)*xm_per_pix
    r.line_base_pos = (rightRoadPosition-640)*xm_per_pix

    if distFromCenterOfLane < 0.0:
        centerDistStr = 'Distance from the center of lane (to left) = ' + '%s' % float('%.1g' % np.absolute(distFromCenterOfLane)) + ' m'
    else:
        centerDistStr = 'Distance from the center of lane (to right) = ' + '%s' % float('%.1g' % np.absolute(distFromCenterOfLane)) + ' m'
        
    #####Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = (1 + (2*l.current_fit[0]*y_eval + l.current_fit[1])**2)**(3/2)/(2*l.current_fit[0])  ## Implement the calculation of the left line here
    right_curverad = (1 + (2*r.current_fit[0]*y_eval + r.current_fit[1])**2)**(3/2)/(2*r.current_fit[0])  ## Implement the calculation of the right line here
    #print('Left radius of curvature = ', left_curverad)
    #print('Right radius of curvature = ', right_curverad)
    RadCurve = (left_curverad + right_curverad)//2
    l.radius_of_curvature = left_curverad
    r.radius_of_curvature = right_curverad
    
    if Frame%5==0:
        if RadCurve < 0.0 and RadCurve > -3500.0:
            RadCurveStr = 'Radius of Curvature = ' + str(np.absolute(RadCurve)) + 'm and turning left' 
        elif RadCurve > 0.0 and RadCurve < 3500.0:
            RadCurveStr = 'Radius of Curvature = ' + str(np.absolute(RadCurve)) + 'm and turning right' 
        else:
            RadCurveStr = 'Radius of Curvature is more than 3500.0 m and driving almost straight'           
    
    topViewRoad = getRoad(l.bestx,r.bestx,ploty)
    img_size = (topViewRoad.shape[1],topViewRoad.shape[0])
    perspectiveImgRoad = cv2.warpPerspective(topViewRoad,Minv,img_size,flags=cv2.INTER_LINEAR)
    perspectiveImgRoad[0:img_size[0]//3,:] = 0
    result = cv2.addWeighted(img, 1, perspectiveImgRoad, 0.3, 0)
    
    Frame = Frame + 1
    start_time = time.time()
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText1 = (10,70)
    bottomLeftCornerOfText2 = (10,100)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    
    cv2.putText(result,RadCurveStr,bottomLeftCornerOfText1, font, fontScale,fontColor,lineType)
    cv2.putText(result,centerDistStr,bottomLeftCornerOfText2, font, fontScale,fontColor,lineType)
     
    return result

def roadMarkingPipeline(img):
    
    #-------------------------------
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    #-------------------------------
    sxbinary,s_binary = sobelFilteredImg(undist)
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary == 1) | (s_binary == 1)] = 1    
    #-------------------------------
    topViewImg = getTopView(combined_binary)       
    #-------------------------------
    fit_polynomial(topViewImg)
    # Generate x and y values for plotting
    ploty = np.linspace(0, topViewImg.shape[0]-1, topViewImg.shape[0] )         
    #-------------------------------    
    markedRoad = getFinalMarkedRoad(ploty,Minv,undist)   
    #-------------------------------     
    return markedRoad
#**********************************************************************************
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


print(os.listdir("../test_videos/"))
frame_size = [720,1280]
RadCurveStr = ''
# Perspective transformation matrix for the given camera and its orientation
M =  np.asarray([[-6.08238872e-01, -1.52810143e+00,  1.02464197e+03], 
      [-2.99760217e-15, -1.53702753e+00,  6.93893282e+02], 
      [-9.10729825e-18, -2.40925787e-03,  1.00000000e+00]])
# Inverse Perspective transformation matrix for the given camera and its orientation
Minv =  np.asarray([[ 1.44125809e-01, -1.00603865e+00,  5.50406105e+02],
 [-7.54951657e-15, -6.50606434e-01,  4.51451434e+02],
 [-1.38777878e-17, -1.56747867e-03,  1.00000000e+00]])
# Number of frames over which averaging is done for smoothing the data
recent_save = 6
# Left and Right instances of the class Line to hold information about the detected lanes.
l = Line()
r = Line()
Frame = 1
# Initialization of some of the variables in the class instances
l.detected = False
r.detected = False
l.current_fit = np.array([-9999,-9999,-9999])
r.current_fit = np.array([-9999,-9999,-9999])
l.recent_xfitted = list(np.zeros((recent_save,frame_size[0])))
l.bestx = np.mean(l.recent_xfitted,axis=0)
r.recent_xfitted = list(np.zeros((recent_save,frame_size[0])))
r.bestx = np.mean(r.recent_xfitted,axis=0)
l.recent_nfit = list(np.zeros((recent_save,3)))
l.best_fit = np.mean(l.recent_nfit,axis=0)
r.recent_nfit = list(np.zeros((recent_save,3)))
r.best_fit = np.mean(r.recent_nfit,axis=0)
RadCurve = 999999

# Reding th evideo file for lane detection
folderString = '../test_videos/'
fileString = 'project_video.mp4'
#fileString = 'challenge_video.mp4'
Video_output_file = '../output_videos/' + fileString
fname = folderString + fileString

clip1 = VideoFileClip(fname).subclip(0,50)
roadWithMarker_clip = clip1.fl_image(roadMarkingPipeline)
roadWithMarker_clip.write_videofile(Video_output_file, audio=False)


