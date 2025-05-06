#Use tf2onnx
python -m tf2onnx.convert --saved-model /home/agus/CODE/python/vehicle/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model --output model.onnx --opset 13


'''
NOT WORKING
import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import numpy as np
import cv2
import sys
import os

def check_gpu():
    print("Checking GPU availability...\n")
    
    # Check TensorFlow GPU
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU devices found for TensorFlow!")
        sys.exit(1)
    
    print(f"✅ Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu}")
    
    # CUDA build check
    if tf.test.is_built_with_cuda():
        print("✅ TensorFlow is built with CUDA support")
    else:
        print("⚠️ TensorFlow is NOT built with CUDA support")
    
    # Try matrix multiplication on GPU
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("✅ Successfully performed matrix multiplication on GPU\n")
    except Exception as e:
        print(f"❌ ERROR: Could not perform operations on GPU: {e}")
        sys.exit(1)

def convert_saved_model_to_onnx(saved_model_path, output_onnx_path):
    print("🔄 Loading TensorFlow model...")
    
    # Load the saved model
    model = tf.saved_model.load(saved_model_path)
    concrete_func = model.signatures['serving_default']
    
    # Get input shape
    input_shape = concrete_func.inputs[0].shape
    print(f"ℹ️ Model input shape: {input_shape}")
    
    # Output names
    output_names = list(concrete_func.structured_outputs.keys())
    print(f"ℹ️ Output names: {output_names}")
    
    # Input signature
    input_signature = [tf.TensorSpec([1, 320, 320, 3], tf.uint8, name="input_tensor")]
    
    # Convert to ONNX
    print("🚀 Converting to ONNX format...")
    model_proto, _ = tf2onnx.convert.from_function(
        concrete_func,
        input_signature=input_signature,
        opset=13,
        output_path=output_onnx_path
    )
    
    print(f"✅ Model successfully saved to '{output_onnx_path}'\n")
    return model_proto

def verify_model(onnx_model_path, test_image):
    print("🧪 Verifying ONNX model...")
    
    try:
        session = ort.InferenceSession(onnx_model_path)
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        sys.exit(1)
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"ℹ️ Input name: {input_name}")
    print(f"ℹ️ Input shape: {input_shape}")
    
    # Run inference
    print("\n⚙️ Running inference with dummy image...")
    outputs = session.run(None, {input_name: test_image})
    
    print("\n📤 Output shapes:")
    for i, output in enumerate(outputs):
        print(f"  Output {i}: {output.shape}")
    
    return outputs

def main():
    check_gpu()
    
    saved_model_path = "/home/agus/CODE/python/vehicle/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
    output_onnx_path = "model.onnx"
    
    # Convert and save ONNX
    convert_saved_model_to_onnx(saved_model_path, output_onnx_path)
    
    # Dummy image (black)
    test_image = np.zeros((1, 320, 320, 3), dtype=np.uint8)
    
    # Run verification
    verify_model(output_onnx_path, test_image)
    
    print("\n✅ Conversion and verification complete!")

if __name__ == "__main__":
    main()


'''
