import tensorflow as tf
import cv2
import time  # Tambahkan modul time untuk menghitung FPS

# 1. Load SavedModel (folder, bukan file .pb)
model = tf.saved_model.load(
    './ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
)

# 2. Ambil signature default untuk inferensi
infer = model.signatures['serving_default']

# 3. Buka video
cap = cv2.VideoCapture('./video.mp4')
if not cap.isOpened():
    raise IOError("Tidak bisa membuka video ./video.mp4")

# Variabel untuk menghitung FPS
prev_time = time.time()

# 4. Loop per-frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # selesai jika frame habis

    # 5. Pre-process: ke tensor dan tambahkan batch dim
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # 6. Inferensi
    outputs = infer(input_tensor)

    # 7. Ambil hasil dan gambar bounding box
    boxes   = outputs['detection_boxes'].numpy()[0]
    scores  = outputs['detection_scores'].numpy()[0]
    classes = outputs['detection_classes'].numpy()[0]

    h, w, _ = frame.shape
    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.5:
            continue

        y1, x1, y2, x2 = box
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))

        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(frame,
                    f'{int(cls)}:{score:.2f}',
                    (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Tampilkan FPS di frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 8. Tampilkan
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Bersihkan
cap.release()
cv2.destroyAllWindows()


