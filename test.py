import numpy as np
import cv2
import math
import matplotlib.image as mpimg
import collections
from matplotlib import pyplot as plt
from calibration import calib,undistort
from collections import deque

input_name='./test_video.mp4'
mtx,dist=calib()
handle_angle = 0
red_color = (0, 0, 255)

def grid_score(frame, left_high, right_low):
    score = np.sum(frame[left_high[1]:right_low[1],left_high[0]:right_low[0]])
    return score

def sliding_window(frame):
    grid_x = 32
    grid_y = 10

    margin_x = frame.shape[1]/grid_x
    margin_y = frame.shape[0]/grid_y

    frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    histogram = np.sum(frame[:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    leftx_current = int(leftx_base/margin_x)
    rightx_current = int(rightx_base/margin_x)

    left_grid = 3
    right_grid = 3

    left_line = []
    right_line = []


    for grid in range(grid_y):
        left_score = []
        right_score = []

        left_point = (int(margin_x*leftx_current+margin_x/2), int(margin_y*(grid_y-grid-1)+margin_y/2))
        right_point = (int(margin_x*rightx_current+margin_x/2), int(margin_y*(grid_y-grid-1)+margin_y/2))
        left_line.append(left_point)
        right_line.append(right_point)
        cv2.line(frame_color, left_point, left_point, (0,0,255), 5)
        cv2.line(frame_color, right_point, right_point, (255,0,0), 5)

        for left in range(left_grid):
            left_p1 = (int(margin_x*(leftx_current+left-int(left_grid/2))),int(margin_y*(grid_y-grid-1)))
            left_p2 = (int(margin_x*(leftx_current+left-int(left_grid/2)+1)),int(margin_y*(grid_y-grid)))
            left_score.append(grid_score(frame,left_p1,left_p2))
            cv2.rectangle(frame_color, left_p1, left_p2, (0,0,255), 2)

        if np.max(left_score) == 0:
            left_grid = 5
        else:
            leftx_current = leftx_current-int(left_grid/2)+np.argmax(left_score)
            left_grid = 3

        for right in range(right_grid):
            right_p1 = (int(margin_x*(rightx_current+right-int(right_grid/2))),int(margin_y*(grid_y-grid-1)))
            right_p2 = (int(margin_x*(rightx_current+right-int(right_grid/2)+1)),int(margin_y*(grid_y-grid)))
            right_score.append(grid_score(frame,right_p1,right_p2))
            cv2.rectangle(frame_color, right_p1, right_p2, (255,0,0), 2)

        if np.max(right_score) == 0:
            right_grid = 5
        else:
            rightx_current = rightx_current-int(right_grid/2)+np.argmax(right_score)
            right_grid = 3
        
    return left_line, right_line, frame_color

def sobel_filter(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray scale 변환

    #sobel x,y filtering for gradient detection ====
    # sobel 결과에 절대값을 적용하고 값 범위를 unsigned int로 변경 
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize = 3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize = 3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
    #sobel x,y = sobel x + sobel y 
    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
    # detecting white mark and yellow mark
    # first,
    _, white_line = cv2.threshold(img_sobel, 160, 255, cv2.THRESH_BINARY)
    return white_line

def top_view(frame):
    pts1 = np.float32([[245, 240], [395, 240], [0, 360], [640, 360]])
    pts2 = np.float32([[0, 0], [640, 0], [0, 360], [640, 360]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    matrix_inv = cv2.getPerspectiveTransform(pts2, pts1)
    frame = cv2.warpPerspective(frame, matrix_inv, (640, 360))
    return frame

def bird_eye_view(frame):
    # pts1, pts2 is ROI
    pts1 = np.float32([[245, 240], [395, 240], [0, 360], [640, 360]])
    pts2 = np.float32([[0, 0], [640, 0], [0, 360], [640, 360]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    matrix_inv = cv2.getPerspectiveTransform(pts2, pts1)
    frame = cv2.warpPerspective(frame, matrix, (640, 360))
    return frame

def predict_line(frame, angle):
    h = 0
    radius = 0
    if angle > 0 :
        th = math.radians(angle)
        radius = 200/math.tan(th)+100
        circle_center = (320+int(radius), 400)
        frame = cv2.ellipse(frame, circle_center, (abs(int(radius)), abs(int(radius))), 0, -180, 0, red_color, 3)
    elif angle < 0:
        th = math.radians(angle)
        radius = 200/math.tan(th)-100
        circle_center = (320+int(radius), 400)
        frame = cv2.ellipse(frame, circle_center, (abs(int(radius)), abs(int(radius))), 0, -180, 0, red_color, 3)
    else :
        frame = cv2.line(frame, (320,360), (320,0), red_color, 3)
    return frame

def yellow_and_white_filter(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 160 #130
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20,20,100])
    upper_yellow = np.array([32,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    
    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    
    return white_image, yellow_image, image2

if __name__=='__main__':
    #cap=cv2.VideoCapture(0)
    cap1=cv2.VideoCapture(input_name)
    while(cap1.isOpened()):

        # step0. Initializing ====
        ret,frame=cap1.read()
        frame=cv2.resize(frame,(640,360))


        # step1. Calibration image ====
        undist_img=undistort(frame, mtx, dist)
        cv2.imshow('1. original image', undist_img)


        # step2. Bird-eye-view (Perspective transformation) ====
        transformed_img = bird_eye_view(undist_img)
        #cv2.imshow('2. bird eye view image', transformed_img)


        # step3-1. Sobel filtering ====
        sobel_filtered_img = sobel_filter(transformed_img)
        #cv2.imshow('3-1. Sobel filtered image', sobel_filtered_img)


        # step3-2. Yellow and White color filtering ====
        white_filtered, yellow_filtered, color_filtered_img = yellow_and_white_filter(transformed_img)
        color_filtered_img = cv2.cvtColor(color_filtered_img, cv2.COLOR_BGR2GRAY)
        _, color_filtered_img = cv2.threshold(color_filtered_img, 1, 255, cv2.THRESH_BINARY)
        #cv2.imshow('3-2. white_filtered image', white_filtered)
        #cv2.imshow('3-2. yellow_filtered image', yellow_filtered)
        #cv2.imshow('3-2. Yellow and White filtered imagee', color_filtered_img)


        # step3-3. Final Filtering
        filtered_img = cv2.bitwise_or(sobel_filtered_img, color_filtered_img)
        #cv2.imshow('3-3. Filtered image', filtered_img)


        # step4. Sliding Window
        left_line, right_line, sliwin_img = sliding_window(filtered_img)
        cv2.imshow('4. Sliding Window', sliwin_img)

        # hidden step. Predict driving line from handle angle
        #window_searched_img = predict_line(window_searched_img, handle_angle)
        #cv2.imshow('hidden. Predict line', window_searched_img)


        # step5. Reverse perspective transform
        original = top_view(sliwin_img)
        cv2.imshow('5. Reverse Perspective Transform', original)


        #곡률구하고 하면 될 것 같습니다 %%
        key = cv2.waitKeyEx(30)
        if key == 0x250000 and handle_angle > -45:
            handle_angle = handle_angle - 1
        elif key == 0x270000 and handle_angle < 45:
            handle_angle = handle_angle + 1

cap1.release()
cv2.destroyAllWindows()