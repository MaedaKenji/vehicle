from tensorflow.python.compiler.tensorrt import trt_convert as trt
 
# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir='ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model',
   precision_mode=trt.TrtPrecisionMode.FP32
)
 
# Convert the model into TRT compatible segments
trt_func = converter.convert()
converter.summary()

MAX_BATCH_SIZE=128
def input_fn():
   batch_size = MAX_BATCH_SIZE
   x = x_test[0:batch_size, :]
   yield [x]
 
converter.build(input_fn=input_fn)

OUTPUT_SAVED_MODEL_DIR="./tftrt_saved_model"
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)

# Get batches of test data and run inference through them
infer_batch_size = MAX_BATCH_SIZE // 2
for i in range(10):
   print(f"Step: {i}")
  
   start_idx = i * infer_batch_size
   end_idx   = (i + 1) * infer_batch_size
   x = x_test[start_idx:end_idx, :]
 
   trt_func(x)