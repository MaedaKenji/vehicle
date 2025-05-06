import cv2
import onnxruntime as ort
import numpy as np
import time  # Tambahkan modul time untuk menghitung FPS

# Load ONNX model
onnx_model_path = "model_jelek.onnx"
session = ort.InferenceSession(onnx_model_path)
input_meta = session.get_inputs()[0]
input_name = input_meta.name

# COCO dataset class labels
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Ambil ukuran input dengan fallback
input_shape = input_meta.shape
try:
    height = int(input_shape[2])
    width = int(input_shape[3])
except Exception:
    # fallback default if shape not known
    height, width = 224, 224

print(f"Using input size: {width}x{height}")

def preprocess(frame):
    img = cv2.resize(frame, (width, height))  # (H, W, C)
    img = np.expand_dims(img, axis=0)         # (1, H, W, C)
    img = img.astype(np.uint8)                # Pastikan sesuai tipe model
    return img

# Open video
cap = cv2.VideoCapture("video.mp4")

# Variabel untuk menghitung rata-rata FPS
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_tensor = preprocess(frame)

    # Jalankan inferensi
    outputs = session.run(None, {input_name: input_tensor})

    # Ambil hasil deteksi
    detection_boxes = outputs[0]  # Asumsikan output pertama adalah bounding box
    detection_scores = outputs[1]  # Asumsikan output kedua adalah skor
    detection_classes = outputs[2]  # Asumsikan output ketiga adalah kelas
    
    print("\nDetection Debug Info:")
    print("detection_boxes shape:", detection_boxes.shape)
    print("detection_scores shape:", detection_scores.shape)
    print("detection_classes shape:", detection_classes.shape)
    print("First detection box:", detection_boxes[0])
    print("First detection score:", detection_scores[0])
    print("First detection class:", detection_classes[0])

    # Reshape detection_boxes to (num_detections, 4)
    detection_boxes = detection_boxes.reshape(-1, 4)
    print("detection_boxes shape after reshape:", detection_boxes.shape)

    h, w, _ = frame.shape
    print(f"Frame dimensions: {w}x{h}")
    
    # Find the maximum values in the detection boxes to normalize
    max_x = np.max(detection_boxes[:, [1, 3]])  # max of x1 and x2
    max_y = np.max(detection_boxes[:, [0, 2]])  # max of y1 and y2
    print(f"Max coordinates - x: {max_x}, y: {max_y}")
    
    for i in range(len(detection_boxes)):  # Iterate over the number of detections
        score_item = detection_scores[0][i]
        score = float(score_item.max())  # or score_item[0] if only one class
        cls = score_item.argmax()  # Get the class with the highest score
        
        if score > 0.5:  # Threshold untuk menampilkan deteksi
            # Get raw coordinates
            y1, x1, y2, x2 = detection_boxes[i]
            print(f"\nDetection {i}:")
            print(f"Raw coordinates: y1={y1}, x1={x1}, y2={y2}, x2={x2}")
            
            # Normalize coordinates to 0-1 range
            x1 = x1 / max_x
            y1 = y1 / max_y
            x2 = x2 / max_x
            y2 = y2 / max_y
            
            # Convert to pixel coordinates
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            print(f"Pixel coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            print(f"Final coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"Class: {cls}, Score: {score}")
            
            # Choose color based on class (you can customize these colors)
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Red, Blue, Yellow
            color = colors[cls % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            class_name = COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else f"Class {int(cls)}"
            label = f"{class_name}: {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Hitung FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Tampilkan rata-rata FPS di frame
    cv2.putText(frame, f'Avg FPS: {avg_fps:.2f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Tampilkan frame
    cv2.imshow("ONNX Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
   