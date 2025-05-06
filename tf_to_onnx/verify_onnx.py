import onnxruntime as ort
import numpy as np
import os

# Inisialisasi sesi inferensi
dir = os.path.dirname(__file__)
session = ort.InferenceSession(os.path.join(dir, "model.onnx"))

# Dapatkan nama input
input_name = session.get_inputs()[0].name

# Buat input dummy sesuai dengan bentuk input model
dummy_input = np.zeros((1, 320, 320, 3), dtype=np.uint8)

# Jalankan inferensi
outputs = session.run(None, {input_name: dummy_input})

# Tampilkan output
for i, output in enumerate(outputs):
    print(f"Output {i}: {output.shape}")
