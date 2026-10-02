from ultralytics import YOLO
import cv2 # OpenCV for video streaming, display, writing also read detail of given video

model = YOLO("yolov8m.pt")# calling pre-trained yolov8 cnn architecture m -> stands for medium variant,
                          # pt -> store weights of that pre-trained model

cap = cv2.VideoCapture(0) # open default webcam , we can also pass video "f1.mp4"

# FOURCC = “Four Character Code , it’s a 4-letter code that tells the computer which codec was used to compress the video
# codec in compress video data
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# read fps (capture each frame)from video and store it
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# read height and width from video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# use all the info collected above to make new video with file name output
out = cv2.VideoWriter("output.mp4",fourcc , fps ,(width,height))

# if the video file is open then read frame → actual image array (NumPy array)
# ret → Boolean (True if frame read successfully)
while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break
    # result store the predicted class from each frame which have confidence >= 40%
    result = model.predict(frame,stream=True,conf=0.4)
    # result contain many frame over here each frame get plotted with their predicted class
    for r in result:
        annotated_frame = r.plot()
    # finally all plotted frame get combine in one video
    out.write(annotated_frame)
    # as video get ready to play , cv2.imshow play the video with title YOLOv8 Detection
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) == 27: # the video will run until it ends or we press esc
        break

# Clean and proper cleanup — prevents camera lock or video file corruption.
cap.release()
out.release()
cv2.destroyAllWindows()
