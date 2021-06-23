# opencv_lane_detection

Lane Detection Code using OpenCV-Python


## Install libraries
```
pip install -r requirements.txt
```


## Start
```
python detection.py
```


# Algorithm
1. Camera Reading / Video Reading
2. Perspective Transformation (Bird's eye View)
3. Filtering
4. Sliding Window
5. Reverse Perspective Transformation (Top View)

## Perspective Transformation
```python
def bird_eye_view(frame):
    # set ROI
    pts = np.float32([[0, 0], [x_size, 0], [0, y_size], [x_size, y_size]])
    matrix = cv2.getPerspectiveTransform(RoI, pts)
    matrix_inv = cv2.getPerspectiveTransform(pts, RoI)
    frame = cv2.warpPerspective(frame, matrix, (x_size, y_size))
    return frame
```
## Filtering
### Edge Filtering
```python
def scharr_filter(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scharr x,y filtering for gradient detection
    img_scharr_x = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
    img_scharr_x = cv2.convertScaleAbs(img_scharr_x)
    img_scharr_y = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
    img_scharr_y = cv2.convertScaleAbs(img_scharr_y)

    # scharr x, y = scharr x + scharr y
    img_scharr = cv2.addWeighted(img_scharr_x, 1, img_scharr_y, 1, 0)

    _, white_line = cv2.threshold(img_scharr, 150, 255, cv2.THRESH_BINARY)
    return white_line
```
### Color Filtering
```python
def yellow_and_white_filter(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 180 #130
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 40, 100])
    upper_yellow = np.array([23, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    
    # Combine the two above images
    out = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    
    return white_image, yellow_image, out
```
#### White Filtering
```python
    # Filter white pixels
    white_threshold = 180 #130
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
```
#### Yellow Filtering
```python
    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 40, 100])
    upper_yellow = np.array([23, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
```
### Thickening
```python
# setp3-3. thickening detected lane
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thickened_color_filtered_img = cv2.dilate(color_filtered_img, kernel)
cv2.imshow('3-3. thickened', thickened_color_filtered_img)
```
### Median Blur
```python
# step3-5. Median blur
median_img=cv2.medianBlur(filtered_img, 5)
```
## Sliding Window
```python
def sliding_window(frame, search_point):
    frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # grid number
    grid_x = 20
    grid_y = 10

    # grid number for searching
    left_grid = 3
    right_grid = 3

    # search starting point of next frame
    next_search = [0,0]

    # grid size
    margin_x = frame.shape[1] / grid_x
    margin_y = frame.shape[0] / grid_y

    # histogram of white pixel to get search starting point of left, right lane
    histogram = np.sum(frame[:,:], axis=0)

    # get midpoint of image and it become boundary of left and right lane
    midpoint = int(histogram.shape[0]/2)

    # search starting point from histogram
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # if there are search points from previous frame,
    # initialize starting point
    if search_point[0] != 0 and search_point[0] < midpoint:
        leftx_base = search_point[0]
    if search_point[1] != 0 and midpoint < search_point[1]: 
        rightx_base = search_point[1]
    
    # from leftx_base and rightx_base, get index of searching window
    leftx_current = int(leftx_base/margin_x)
    rightx_current = int(rightx_base/margin_x)

    # list of lane point
    left_line = []
    right_line = []

    # start searching lane from bottom of image to top
    for grid in range(grid_y):
        # white pixels of each window are calculated to score
        left_score = []
        right_score = []

        # first, assume that there are no lane point
        left_point_exist = False
        right_point_exist = False


        # search white pixel of left side of image
        for left in range(left_grid):
            left_p1 = (int(margin_x * (leftx_current + left - int(left_grid / 2))), int(margin_y * (grid_y - grid - 1)))
            left_p2 = (int(margin_x * (leftx_current + left - int(left_grid / 2) + 1)), int(margin_y * (grid_y - grid)))
            left_score.append(grid_score(frame, left_p1, left_p2))      # calculate white pixel score of each window and append it to list
            cv2.rectangle(frame_color, left_p1, left_p2, red_color, 2)  # draw window
        
        # if there are no white pixel in left side
        if np.max(left_score) == 0:
            left_grid = 5   # set number of searching window of next frame to 5
        # if there are white pixel in left side
        else:
            leftx_current = leftx_current - int(left_grid / 2) + np.argmax(left_score)  # set searching point of upper window
            left_grid = 3                                                               # set number of searching window of next frame to 3
            left_point_exist = True


        # search white pixel of right side of image
        for right in range(right_grid):
            right_p1 = (int(margin_x * (rightx_current + right - int(right_grid / 2))), int(margin_y * (grid_y - grid - 1)))
            right_p2 = (int(margin_x * (rightx_current + right - int(right_grid / 2) + 1)), int(margin_y * (grid_y - grid)))
            right_score.append(grid_score(frame, right_p1, right_p2))       # calculate white pixel score of each window and append it to list
            cv2.rectangle(frame_color, right_p1, right_p2, blue_color, 2)   # draw window
        
        # if there are no white pixel in right side
        if np.max(right_score) == 0:
            right_grid = 5  # set number of searching window of next frame to 5
        # if there are white pixel in right side
        else:
            rightx_current = rightx_current - int(right_grid / 2) + np.argmax(right_score)  # set searching point of upper window
            right_grid = 3                                                                  # set number of searching window of next frame to 3
            right_point_exist = True
        
        # set left and right lane points
        left_point = (int(margin_x * leftx_current + margin_x / 2), int(margin_y * (grid_y - grid - 1) + margin_y / 2))
        right_point = (int(margin_x * rightx_current + margin_x / 2), int(margin_y * (grid_y - grid - 1) + margin_y / 2))

        # if right_point and left point are close each other, choice one point that have more points before
        if (right_point[0] - left_point[0]) < 200:
            if len(left_line) < len(right_line):
                left_point_exist = False
            elif len(left_line) > len(right_line):
                right_point_exist = False

        if left_point_exist == True:
            # draw left point
            cv2.line(frame_color, left_point, left_point, red_color, 10)
            if right_point_exist == True:
                # left point O, right point O
                cv2.line(frame_color, right_point, right_point, blue_color, 10) # draw right point
                # if calculated left point is in range
                if right_point[0] < x_size:
                    right_line.append(right_point)  # append it to list
            else:
                # left point O, right point X
                # assume that left lane is curved lane, and reinforce searching of left lane
                left_grid = 5
            # if calculated left point is in range
            if left_point[0] > 0:
                left_line.append(left_point)    # append it to list
        else:
            if right_point_exist == True:
                # left point X, right point O
                # assume that right lane is curved lane, and reinforce searching of right lane
                right_grid = 5
                cv2.line(frame_color, right_point, right_point, blue_color, 10) # draw right point
                # if calculated right point is in range
                if right_point[0] < x_size:
                    right_line.append(right_point)  # append it to list
        
        # lane points of second window from bottom of image are saved to help next frame to set searching point
        if grid == 1:
            if left_point_exist == True:
                next_search[0] = left_point[0]
            if right_point_exist == True:
                next_search[1] = right_point[0]

    return frame_color, left_line, right_line, next_search
```
## Reverse Perspective Transformation
```python
def top_view(frame):
    # set RoI
    pts = np.float32([[0, 0], [x_size, 0], [0, y_size], [x_size, y_size]])
    matrix = cv2.getPerspectiveTransform(RoI, pts)
    matrix_inv = cv2.getPerspectiveTransform(pts, RoI)
    frame = cv2.warpPerspective(frame, matrix_inv, (x_size, y_size))
    return frame
```
# Result
<img width="80%" src="https://user-images.githubusercontent.com/69493518/123046806-82f02c80-d437-11eb-82d4-d92f51752350.gif"/>
