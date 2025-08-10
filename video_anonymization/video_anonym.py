import cv2
import os
from PIL import Image
import numpy as np
import face_recognition

# === Load synthetic faces ===
synthetic_faces = []
face_dir = r'C:\Users\ACER\Desktop\face_anonymization_gan\face_generation\generated_faces'

print("📸 Loading synthetic faces...")
for img_name in sorted(os.listdir(face_dir)):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(face_dir, img_name)
    img = Image.open(img_path).convert("RGB").resize((100, 100))
    synthetic_faces.append(np.array(img))

print(f"✅ Loaded {len(synthetic_faces)} synthetic faces.")

# === Open video ===
video_path = r'C:\Users\ACER\Desktop\face_anonymization_gan\video_anonymization\input_video.mp4'
out_path = r'C:\Users\ACER\Desktop\face_anonymization_gan\video_anonymization\output_video.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Resize output to reduce memory if needed ===
output_width = 1280
output_height = int(height * (output_width / width))

out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

print("🎞️ Processing video...")

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
        break

    print(f"📷 Reading frame {frame_id}...")

    try:
        # Resize frame for faster detection
        process_width = 640
        process_height = int(height * (process_width / width))
        frame_resized = cv2.resize(frame, (process_width, process_height))
        rgb_frame = frame_resized[:, :, ::-1]

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        print(f"👀 Detected {len(face_locations)} face(s)")

        # Scale back to original frame
        scale_x = width / process_width
        scale_y = height / process_height

        for i, (top, right, bottom, left) in enumerate(face_locations):
            top = int(top * scale_y)
            right = int(right * scale_x)
            bottom = int(bottom * scale_y)
            left = int(left * scale_x)

            if top < 0 or left < 0 or bottom > frame.shape[0] or right > frame.shape[1]:
                continue

            synthetic_face = synthetic_faces[i % len(synthetic_faces)]
            resized_face = cv2.resize(synthetic_face, (right - left, bottom - top))
            frame[top:bottom, left:right] = resized_face

        # Resize to output size and write
        output_frame = cv2.resize(frame, (output_width, output_height))
        out.write(output_frame)

        frame_id += 1
        if frame_id % 10 == 0:
            print(f"➡️ Written frame {frame_id}")

    except Exception as e:
        print(f"❌ Error in frame {frame_id}: {e}")
        break

cap.release()
out.release()
print("✅ Finished. Check output_video.mp4")
