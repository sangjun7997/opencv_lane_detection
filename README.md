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

### Color Filtering

### Thickening

### Median Blur

## Sliding Window

## Reverse Perspective Transformation
