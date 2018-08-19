import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
#%matplotlib qt

VF_IR_Bullet_8mm = False
Udacity = True
GoPro = False

# For using a pre-calibrated camera, set this flag to False. For first time calibration, set it to True.
CameraCalibrateFlag = False
# For using an already undistorted images, set this flag to False. For undistorting for the first time, set it to True.
undistortFlag = False
# For using images that have already been converted to color-binary, set this flag to False. 
# For creating the color-binaries for the first time, set it to True.
colorBinaryFlag = False

getTopViewFlag = False

getPolyfitFlag = False

getWithRoadMarkersFlag = True 

#**********************************************************************************
def undistortImage(folderString,fileString,mtx,dist):
    images = glob.glob(folderString + fileString)   # VF_IR_Bullet_8mmFL Database
    for fname in images:
        filenameOnly = fname.split('\\')[-1]
        fname = folderString + filenameOnly
        #print(fname)
        img = cv2.imread(fname)
        undist = cv2.undistort(img,mtx,dist,None,mtx)        
        undistFilesName = folderString + 'undistImages/' + filenameOnly 
        cv2.imwrite(undistFilesName,undist)      
    return

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
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
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary,sxbinary,s_binary

def getColorBinary(folderString,fileString): 
    images = glob.glob(folderString + fileString) 
    for fname in images:
        filenameOnly = fname.split('\\')[-1]
        imageFile = folderString + filenameOnly
        #print(fname)
        image = mpimg.imread(imageFile)
        colorBinary,sxbinary,s_binary = pipeline(image)
        colorBinaryFilesName = folderString + 'colorBinaryImages/' + filenameOnly 
        cv2.imwrite(colorBinaryFilesName,colorBinary)
        
        # Combine the two binary thresholds
        #sxbinary = colorBinary[:,:,1]
        #s_binary = colorBinary[:,:,2]
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(sxbinary == 1) | (s_binary == 1)] = 1        
        combinedBinaryFilesName = folderString + 'combBinImg/' + filenameOnly 
        #print(combinedBinaryFilesName)
        cv2.imwrite(combinedBinaryFilesName,combined_binary*255)

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

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
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
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

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    #plt.imshow(binary_warped)
    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit

def getRoad(img,left_fitx,right_fitx,ploty):
    road = np.zeros_like(img)
    #plt.imshow(img)
    top_left = [left_fitx[0],0]
    top_right = [right_fitx[0],0]
    bottom_left = [left_fitx[-1],719]
    bottom_right = [right_fitx[-1],719]
    
       
    #leftx = (np.vstack([left_fitx,ploty]))
    #rightx = np.flipud(np.vstack([right_fitx,ploty]))
    #plt.plot(left_fitx, ploty, color='yellow')
    left = ()
    right = ()
    for i in range(len(left_fitx)):
        qwe = [left_fitx[len(left_fitx)-1-i],ploty[len(left_fitx)-1-i]]
        #print('qwe = ',qwe)
        left = left + (qwe,)
        
    for i in range(len(left_fitx)):
        qwe = [right_fitx[i],ploty[i]]
        right = right + (qwe,)
            
    temp = ()
    #temp = temp + (left)
    temp = temp + (top_left,)
    temp = temp + (top_right,)
    temp = temp + (right)
    temp = temp + (bottom_right,)
    temp = temp + (bottom_left,)
    temp = temp + (left)
    #temp = temp + (top_left,)
    
    triangle = np.array([temp], np.int32)
    cv2.fillConvexPoly(road, triangle, [0, 255, 0])
    result = cv2.addWeighted(img, 1, road, 0.3, 0)
    #plt.imshow(result)
    return result, road

#**********************************************************************************

#..................................................................................
# For using a new camera for which alibration data is not available
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
#.............................................................................................


#..........................................................................
if undistortFlag == True:
    folderString = '../test_images/'
    fileString = 'test*.jpg'
    undistortImage(folderString,fileString,mtx,dist)
#..........................................................................

#..........................................................................
if colorBinaryFlag == True:
    folderString = '../test_images/undistImages/'
    fileString = 'straight_lines*.jpg'
    getColorBinary(folderString,fileString)
#..........................................................................
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def getROI(img):
    temp1 = np.copy(img)
    temp2 = np.copy(img)
    ysize = img.shape[0]
    xsize = img.shape[1]   
    # Define a triangle region of interest 
    # Keep in mind the origin (x=0, y=0) is in the upper left in image processing
    # Note: if you run this code, you'll find these are not sensible values!!
    # But you'll get a chance to play with them soon in a quiz 
   
    left_bottom = (0, ysize-30)
    right_bottom = (xsize, ysize-30)
    left_top = (xsize//2-50, ysize//2)
    right_top = (xsize//2+50, ysize//2)
    
    
    #masked = white_yellow_filter(gray_image)
    
    vertices = np.array([[left_bottom,right_bottom,left_top, right_top ]])
    masked_RoI = region_of_interest(temp1, vertices)
    
    
    return masked_RoI
    

def getTopView(folderString,fileString,FourCorners):
    images = glob.glob(folderString + fileString)   # VF_IR_Bullet_8mmFL Database
    for fname in images:
        filenameOnly = fname.split('\\')[-1]
        fname = folderString + filenameOnly    
        print(fname)
        img = mpimg.imread(fname)
        
        img_size = (img.shape[1],img.shape[0])
        
        #img = getROI(img)
        img_bot_half = img[:img.shape[0]-30,:]
        #plt.imshow(img)        
        #print(img_size)
        src = np.float32(
            [FourCorners[0],
            FourCorners[1],
            FourCorners[2],
            FourCorners[3]])
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        dst = np.float32([[290,324],[990,324],[990,396],[290,396]])    
        M = cv2.getPerspectiveTransform(src,dst)
        Minv = cv2.getPerspectiveTransform(dst,src)
        #print('M = ',M)
        topViewImg = cv2.warpPerspective(img_bot_half,M,img_size,flags=cv2.INTER_LINEAR)
        #topViewImg = cv2.warpPerspective(topViewImg,Minv,img_size,flags=cv2.INTER_LINEAR)
        topViewImgFilesName = folderString + 'top/' + filenameOnly 
        cv2.imwrite(topViewImgFilesName,topViewImg)        
    return M, Minv

#..........................................................................
if getTopViewFlag == True:
    FourCorners = [[541,489],[746,489],[777,511],[511,511]]
    folderString = '../test_images/undistImages/combBinImg/'
    fileString = 'straight_lines*.jpg'
    M, Minv = getTopView(folderString,fileString,FourCorners)
#..........................................................................

def getPolyfit(folderString,fileString):
    images = glob.glob(folderString + fileString)   # VF_IR_Bullet_8mmFL Database
    for fname in images:
        filenameOnly = fname.split('\\')[-1]
        fname = folderString + filenameOnly    
        print(fname)
        binary_warped = mpimg.imread(fname)    
        polyFitImage,left_fit, right_fit = fit_polynomial(binary_warped)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )        
        # Plots the left and right polynomials on the lane lines
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]        

        margin = 5  # NOTE: Keep this in sync with *_fit()
   
        # Draw the lines on the edge image
        #combo = cv2.addWeighted(temp, 1, polyFitImage, 1, 0)         
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension        
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(ploty)*ym_per_pix
        
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))        
        # Draw the lane onto the warped blank image
        window_img = np.zeros_like(polyFitImage)
        #print(np.int_([left_fitx]))
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 255))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 255))
        result = cv2.addWeighted(polyFitImage, 1, window_img, 1, 0)
        
        #margin = 100  # NOTE: Keep this in sync with *_fit()
   
        ## Draw the lines on the edge image
        ##combo = cv2.addWeighted(temp, 1, polyFitImage, 1, 0)         
        ## Define y-value where we want radius of curvature
        ## We'll choose the maximum y-value, corresponding to the bottom of the image
        ## Define conversions in x and y from pixels space to meters
        #ym_per_pix = 30/720 # meters per pixel in y dimension        
        #xm_per_pix = 3.7/700 # meters per pixel in x dimension
        #y_eval = np.max(ploty)*ym_per_pix
        
        #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        #left_line_pts = np.hstack((left_line_window1, left_line_window2))
        #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        #right_line_pts = np.hstack((right_line_window1, right_line_window2))        
        ## Draw the lane onto the warped blank image
        #window_img = np.zeros_like(polyFitImage)
        ##print(np.int_([left_fitx]))
        #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        #result = cv2.addWeighted(result, 1, window_img, 0.4, 0)        
        #plt.imshow(result)
      
        
        ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
        left_curverad = (1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**(3/2)/np.absolute(2*left_fit[0])  ## Implement the calculation of the left line here
        right_curverad = (1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**(3/2)/np.absolute(2*right_fit[0])  ## Implement the calculation of the right line here
        print('Left radius of curvature = ', left_curverad)
        print('Right radius of curvature = ', right_curverad)
        
        polyFitWithRoad,topViewRoad = getRoad(result,left_fitx,right_fitx,ploty)
        
        polyFitImageFilesName = folderString + 'Polyfit/' + filenameOnly 
        
        img_size = (topViewRoad.shape[1],topViewRoad.shape[0])
        perspectiveImgRoad = cv2.warpPerspective(topViewRoad,Minv,img_size,flags=cv2.INTER_LINEAR)
        perspectiveImgRoad[0:img_size[0]//3,:] = 0
        roadPerspectiveFilesName = folderString + 'rdPersp/' + filenameOnly 
        
        
        cv2.imwrite(polyFitImageFilesName,polyFitWithRoad)         
        cv2.imwrite(roadPerspectiveFilesName,perspectiveImgRoad)  

if getPolyfitFlag == True:
    folderString = '../test_images/undistImages/combBinImg/top/'
    fileString = 'straight_lines*.jpg'
    fname = folderString + fileString
    getPolyfit(folderString,fileString)


def getWithRoadMarkers(folderString,fileString):
    images = glob.glob(folderString + fileString)   # VF_IR_Bullet_8mmFL Database
    for fname in images:
        filenameOnly = fname.split('\\')[-1]
        fname = folderString + filenameOnly    
        #print(fname)
        perspRoad = mpimg.imread(fname)     
        img_size = (perspRoad.shape[1],perspRoad.shape[0])
        #np.repeat(perspRoad.reshape(img_size[1], img_size[0], 1), 3, axis=2)
        
        undistFilePath = '../test_images/undistImages/'
        undistFname = undistFilePath + filenameOnly
        #print(undistFname)
        undistImg = cv2.imread(undistFname) 
        result = cv2.addWeighted(undistImg, 1, perspRoad, 0.3, 0)
        
        undistWithRoadMarkFilesName = undistFilePath + 'withRoadMarkers/' + filenameOnly 
        cv2.imwrite(undistWithRoadMarkFilesName,result) 
        
if getWithRoadMarkersFlag == True:
    folderString = '../test_images/undistImages/combBinImg/top/rdPersp/'
    fileString = 'straight_lines*.jpg'
    fname = folderString + fileString
    getWithRoadMarkers(folderString,fileString)








## Plot the result
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))
#f.tight_layout()

#ax1.imshow(image)
#ax1.set_title('Original Image', fontsize=40)

#ax2.imshow(result)
#ax2.set_title('Pipeline Result', fontsize=40)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

