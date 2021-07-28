import argparse
import os
import numpy as np
from skimage import io
from skimage.transform import resize
import time

# Torch
import torch
#import torchvision.models as models

# ONNX: pip install onnx, onnxruntime
try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

# CUDA & TensorRT
import pycuda.driver as cuda 
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX models to TensorRT')
    parser.add_argument('--device', help='cuda or not',
        default='cuda:0')
    parser.add_argument('--onnx_model_path', help='onnx mode path',
        default='./onnx_output.onnx')    
    parser.add_argument('--input_shape', help='input size',
        default=[1, 3, 256, 192])
    parser.add_argument('--tensorrt_engine_path',  help='experted onnx path',
        default='./tensorrt_engine.engine')
    parser.add_argument('--sample_image_path', help='sample image path',
        default='./sample.jpg')
    parser.add_argument('--fp16_mode',  help='fp16 mode of TensorRT',
        default='False')
    parser.add_argument('--max_batch_size', help='The maximum batch size which can be used at execution time, and also the batch size for which the ICudaEngine will be optimized',
        default=1)
    args = string_to_bool(parser.parse_args())

    return args

def string_to_bool(args):
    
    if args.fp16_mode.lower() in ('true'): args.fp16_mode = True
    else: args.fp16_mode = False

    return args
    
def load_image(img_path, size):
    img_raw = io.imread(img_path)
    img_raw = np.rollaxis(img_raw, 2, 0)
    img_resize = resize(img_raw / 255, size, anti_aliasing=True)
    img_resize = img_resize.astype(np.float32)
    return img_resize, img_raw

def build_engine(onnx_model_path, max_batch_size, fp16_mode):
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
    # Create builder, network, parser
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(explicit_batch) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser: 
        builder.max_workspace_size = 1 << 20 # Amount of memory available to the builder when building an optimized engine
        
        builder.max_batch_size = max_batch_size
        if fp16_mode:
            print('fp16 mode')
            builder.fp16_mode = True
        
        with open(onnx_model_path, "rb") as f:
            parser.parse(f.read())
        #network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
        print("network.num_layers", network.num_layers)

        engine = builder.build_cuda_engine(network)
        return engine

def alloc_buf(engine):
    # Host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # Allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    # Create a stream and run inference.
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream

if __name__ == '__main__':
    args = parse_args()

    # Sample image
    img_resize, img_raw = load_image(args.sample_image_path, args.input_shape)
    print(img_resize.shape, img_raw.shape)
    
    # Build TensorRT engine
    tensorrt_engine = build_engine(args.onnx_model_path, args.max_batch_size, args.fp16_mode)
    
    # Save TensorRT engine
    with open(args.tensorrt_engine_path, "wb") as f:
        f.write(tensorrt_engine.serialize())
        
    # Read the engine from the file and deserialize
    with open(args.tensorrt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime: 
        engine = runtime.deserialize_cuda_engine(f.read())    
    context = engine.create_execution_context()
    
    in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
    
    # Transfer input data to the GPU
    cuda.memcpy_htod(in_gpu, img_resize)
    
    # Run inference
    trt_start_time = time.time()
    context.execute(1, [int(in_gpu), int(out_gpu)])
    trt_end_time = time.time()
    
    # Transfer predictions from the GPU to CPU
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    
    # ONNX inference
    onnx_model = onnx.load(args.onnx_model_path)
    
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    sess = rt.InferenceSession(args.onnx_model_path)
    
    onnx_start_time = time.time()
    onnx_result = sess.run(None,
                            {net_feed_input[0]: img_resize
                            })[0]
    onnx_end_time = time.time()
    
    ## Comparision output of TensorRT and output of onnx model

    # Time Efficiency & output
    print('--onnx--')
    print(onnx_result.shape)
    print(np.argmax(onnx_result, axis=1))
    print('time: ', onnx_end_time - onnx_start_time)
    
    print('--tensorrt--')
    print(out_cpu.shape)
    print(np.argmax(out_cpu, axis=0))
    print('time: ', trt_end_time - trt_start_time)
    
    # Comparision
    assert np.allclose(
        onnx_result, out_cpu,
        atol=1.e-5), 'The outputs are different (ONNX and TensorRT)'
    print('The numerical values are same (ONNX and TensorRT)')
