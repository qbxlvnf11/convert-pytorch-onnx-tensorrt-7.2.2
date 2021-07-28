

Description
=============

#### - ONNX (Open Neural Network Exchange)
  - Standard format for expressing machine learning algorithms and models
  - More details about ONNX: https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange

#### - TensorRT
  - NVIDIA SDK for high-performance deep learning inference
  - Deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications
  - More details about TensorRT: https://blog.naver.com/qbxlvnf11/222403199156
  - Setting TensorRT environment using Docker: https://blog.naver.com/qbxlvnf11/222441230287

Contents
=============
#### - Converting Pytorch to onnx
  - Details: https://blog.naver.com/qbxlvnf11/222342675767
  - Export onnx
  - Load onnx
  - Inference onnx
  - Comparision output of onnx and output of Pytorch (same or not)

#### - Converting onnx to TensorRT
  - Build TensorRT engine
  - Save TensorRT engine
  - Load TensorRT engine
  - Inference TensorRT engine
  - Comparision output of TensorRT and output of onnx
  - Comparision of time efficiency

Examples of inferencing ResNet18
=============
#### - Converting Pytorch model to onnx
```
python convert_pytorch_to_onnx.py --load_file false --sample_image_path ./imagenet-mini/train/n12267677/n12267677_6842.JPEG
```
#### - Converting onnx model to TensorRT (TF32)
```
python convert_onnx_to_tensorrt.py --sample_image_path ./imagenet-mini/train/n12267677/n12267677_6842.JPEG
```
#### - Converting onnx model to TensorRT (FP16)
```
python convert_onnx_to_tensorrt.py --sample_image_path ./imagenet-mini/train/n12267677/n12267677_6842.JPEG --fp16_mode true
```

#### - Comparision of time efficiency (ResNet18, inferencing of './imagenet-mini/train/n12267677/n12267677_6842.JPEG')
  - Time efficiency
    - onnx:  0.016895
    - TensorRT (TF32): 0.001304 (about 13 times more efficient than onnx)
    - TensorRT (FP16): 0.000711 (about 24 times more efficient than onnx)
  - There is a risk that the performance of fp16 will be degraded.
  
References
=============

#### - Converting Pytorch models to onnx

https://pytorch.org/docs/stable/onnx.html

#### - TensorRT

https://developer.nvidia.com/tensorrt

#### - TensorRT Engine Builder

https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html

#### - ImageNet 1000 samples

https://www.kaggle.com/ifigotin/imagenetmini-1000

Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com

