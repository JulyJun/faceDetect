import cv2
from ultralytics import YOLO

def apply_mosaic(image, results, scale=0.1):
    for r in results.xyxy[0].cpu().numpy():
        x, y, w, h = map(int, r[:4])
        roi = image[y:y+h, x:x+w]
        roi = cv2.resize(roi, (0, 0), fx=scale, fy=scale)
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = roi
    return image

print("start")
# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the video file
video_path = 0
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.4)

        # Apply mosaic effect over detected objects
        frame = apply_mosaic(frame, results)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference with Mosaic", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
