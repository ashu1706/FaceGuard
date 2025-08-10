import cv2
import face_recognition

print("🔁 Starting script...")

video_path = "C:/Users/ACER/Desktop/face_anonymization_gan/video_anonymization/input_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

print("✅ Video opened.")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"📽 FPS: {fps}, Width: {frame_width}, Height: {frame_height}")

out = cv2.VideoWriter(
    "C:/Users/ACER/Desktop/face_anonymization_gan/video_anonymization/output_video.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠️ End of video or failed to read frame.")
        break

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    print(f"➡️ Frame {frame_count+1}: {len(face_locations)} face(s) detected.")

    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Finished. Check output_video.mp4")

