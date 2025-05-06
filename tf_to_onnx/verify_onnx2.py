import cv2
import os
import time
import numpy as np
import onnxruntime as ort

# Inisialisasi video capture
cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Error: Tidak dapat membuka video.")
    exit()

# Inisialisasi sesi ONNX
dir_path = os.path.dirname(__file__)
session = ort.InferenceSession(os.path.join(dir_path, "model.onnx"))
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, height, width, 3]
height = input_shape[1]
width = input_shape[2]

# Loop untuk membaca frame video
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing frame
    input_frame = cv2.resize(frame, (320, 320))
    input_tensor = np.expand_dims(input_frame, axis=0).astype(np.uint8)

    # Inferensi
    outputs = session.run(None, {input_name: input_tensor})

    # Ekstraksi hasil deteksi
    boxes = outputs[1][0]  # detection_boxes
    classes = outputs[2][0]  # detection_classes
    scores = outputs[4][0]  # detection_scores
    num_detections = int(outputs[5][0])  # num_detections

    # Gambar bounding box
    for i in range(num_detections):
        score = scores[i]
        if score > 0.5:
            box = boxes[i]
            y1, x1, y2, x2 = box
            x1 = int(x1 * frame.shape[1])
            x2 = int(x2 * frame.shape[1])
            y1 = int(y1 * frame.shape[0])
            y2 = int(y2 * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {int(classes[i])}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hitung dan tampilkan FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Tampilkan frame
    cv2.imshow('Frame', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
