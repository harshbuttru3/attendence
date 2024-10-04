import cv2

# Try different indices, usually 0, 1, 2, etc.
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i} is working.")
            cap.release()
            break
        cap.release()
