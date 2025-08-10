import cv2

video_path = "C:/Users/ACER/Desktop/face_anonymization_gan/video_anonymization/input_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed.")
        break

    frame_count += 1
    print(f"Frame {frame_count} read successfully.")

    if frame_count >= 3:  # limit test
        break

cap.release()
print("Done.")
