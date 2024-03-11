import cv2
from ultralytics import YOLO

def draw_roi_rectangles(frame, boxes, rate):
    for box in boxes:
        x, y, w, h = map(int, box)
        w = int(w/2)
        h = int(h/2)
        x = x - w
        y = y - h
        w = w * 2
        h = h * 2
        print("x: " + str(x) + ", y: " + str(y) + ",w: " + str(w) + ",h: " + str(h))
        if x and y:
            roi = frame[abs(y-h):y+h, abs(x-w):x+w]
            print("roi size: " + str(roi.size))
            if roi.size != 0:
                print("enter roi works")
                try:
                    roi = cv2.resize(roi, (w // rate, h // rate), interpolation=cv2.INTER_AREA)
                    roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
                    frame[y:y + h, x:x + w] = roi
                except cv2.error as e:
                    print(f"Error resizing: {e}")
        

def changeVal(event, x, y, flags, param):
    global doMosaic
    if event == cv2.EVENT_LBUTTONDOWN:
        doMosaic = not doMosaic
        print("activate callback")

model = YOLO("best.pt")

video_path = 0
cap = cv2.VideoCapture(video_path)
rate = 15
doMosaic = False
cv2.namedWindow('Mosaic IMG')
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        cv2.putText(frame, str(doMosaic), (0, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 3)
        cv2.setMouseCallback("Mosaic IMG", changeVal)
        results = model.predict(frame, conf=0.4, show=False, verbose=False)
        #results = model.predict("https://www.youtube.com/watch?v=EzhTqWUNUjI",stream=True)
        boxes = results[0].boxes.xywh if len(results) > 0 else []

        if doMosaic == True:
            draw_roi_rectangles(frame, boxes, rate)

        # Display the annotated frame
        cv2.imshow("Mosaic IMG", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()