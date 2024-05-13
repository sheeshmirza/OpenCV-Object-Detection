import cv2

def main():
    # Load the pre-trained classifier for detecting human faces (front side)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    # Open camera for capturing frames
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    # Main loop for capturing and processing frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Convert frame to grayscale for better face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the grayscale frame
            objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
            # Draw rectangles around the detected faces
            for (x, y, w, h) in objects:
                size = max(w, h)
                cv2.rectangle(frame, (x, y), (x+size, y+size), (0, 255, 0), 5)
                # Print coordinates
                print("Object detected at (x={}, y={})".format(x, y))
            # Display the resulting frame
            cv2.imshow('OpenCV Object Detection Project By S. Mirza', frame)
        else:
            print("Error: Unable to read frame")
            break
    # When everything done, release the capture
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()