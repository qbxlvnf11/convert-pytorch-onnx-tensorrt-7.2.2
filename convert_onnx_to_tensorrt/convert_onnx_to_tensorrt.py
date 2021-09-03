import argparse
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from collections import OrderedDict
from PIL import Image
import cv2
import time

# Torch
import torch
from torch import nn
import torchvision.models as models
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

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

TRT_LOGGER = trt.Logger()

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX models to TensorRT')
    parser.add_argument('--device', help='cuda or not',
        default='cuda:0')
    #parser.add_argument('--dynamic_axes', help='dynamic batch input or output',
        #default='False')
    parser.add_argument('--onnx_model_path', help='onnx mode path (not dynamic axes)',
        default='./onnx_output.onnx')    
    parser.add_argument('--batch_size', type=int, help='data batch size',
        default=1)
    parser.add_argument('--img_size', help='input size',
        default=[1, 3, 224, 224])
    #parser.add_argument('--output_shape', help='input size',
        #default=[1, 1000])
    parser.add_argument('--tensorrt_engine_path',  help='experted onnx path',
        default='./tensorrt_engine.engine')
    #parser.add_argument('--sample_image_path', help='sample image path',
        #default='./sample.jpg')
    parser.add_argument('--sample_folder_path', help='sample image folder path',
        default='./imagenet-mini/train')
    parser.add_argument('--fp16_mode',  help='fp16 mode of TensorRT',
        default='False')
    #parser.add_argument('--min_batch_size', type=int, help='The minimum batch size which can be used at execution time, and also the batch size for which the ICudaEngine will be optimized',
        #default=1)
    #parser.add_argument('--max_batch_size', type=int, help='The maximum batch size which can be used at execution time, and also the batch size for which the ICudaEngine will be optimized',
        #default=32)
    args = string_to_bool(parser.parse_args())

    return args

def string_to_bool(args):
    
    if args.fp16_mode.lower() in ('true'): args.fp16_mode = True
    else: args.fp16_mode = False

    return args

def get_transform(img_size):
    options = []
    options.append(transforms.Resize((img_size[2], img_size[3])))
    options.append(transforms.ToTensor())
    #options.append(transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
    transform = transforms.Compose(options)
    return transform
    
'''def load_image(img_path, size):
    img_raw = io.imread(img_path)
    img_raw = np.rollaxis(img_raw, 2, 0)
    img_resize = resize(img_raw / 255, size, anti_aliasing=True)
    img_resize = img_resize.astype(np.float32)
    return img_resize, img_raw'''

def load_image_folder(folder_path, img_size, batch_size):
    transforming = get_transform(img_size)
    dataset = datasets.ImageFolder(folder_path, transform=transforming)
    data_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1)
    data_iter = iter(data_loader)
    torch_images, class_list = next(data_iter)
    print('class:', class_list)
    print('torch_images size:', torch_images.size())
    save_image(torch_images[0], 'sample.png')
    
    return torch_images.cpu().numpy()

#def build_engine(onnx_model_path, min_batch_size, max_batch_size, img_size, output_shape, fp16_mode, dynamic_axes, tensorrt_engine_path):
def build_engine(onnx_model_path, fp16_mode, tensorrt_engine_path):
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # Create builder, config, network, parser, runtime
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(explicit_batch) as network, \
        builder.create_builder_config() as config, \
        trt.OnnxParser(network, TRT_LOGGER) as parser, \
        trt.Runtime(TRT_LOGGER) as runtime:
        
        # tnesorrt version: 21.08
        '''config.max_workspace_size = 1 << 28 # Amount of memory available to the builder when building an optimized engine'''
        builder.max_workspace_size = 1 << 28 # Amount of memory available to the builder when building an optimized engine
        #builder.max_batch_size = max_batch_size
        if fp16_mode:
            print('fp16 mode')
            builder.fp16_mode = True
        
        with open(onnx_model_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
        
        '''if dynamic_axes:
            min_input_shape = (min_batch_size, img_size[1], img_size[2], img_size[3])
            max_input_shape = (max_batch_size, img_size[1], img_size[2], img_size[3])
            min_output_shape = (min_batch_size, output_shape[1])
            max_output_shape = (max_batch_size, output_shape[1])
            print('min_input_shape:', min_input_shape, 'max_input_shape:', max_input_shape)
            print('min_output_shape:', min_output_shape, 'max_output_shape:', max_output_shape)
            
            profile = builder.create_optimization_profile()    
            profile.set_shape("input", min=min_input_shape, opt=max_input_shape, max=max_input_shape)
            profile.set_shape("output", min=min_output_shape, opt=max_output_shape, max=max_output_shape)
            #profile.set_shape("input", (1, 224, 224, 3), (2, 224, 224, 3), (3, 224, 224, 3))
            config.add_optimization_profile(profile)'''
        
        # tnesorrt version: 21.08
        '''plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)'''
        engine = builder.build_cuda_engine(network)
        
        print("Completed creating Engine")
        
        # Save TensorRT engine
        # tnesorrt version: 21.08
        '''with open(tensorrt_engine_path, "wb") as f:
            f.write(plan)'''
        with open(tensorrt_engine_path, "wb") as f:
            f.write(engine.serialize())

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))# * engine.max_batch_size
        #print('batch_size:', engine.max_batch_size, 'alloc size:', size)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

if __name__ == '__main__':
    args = parse_args()

    # Sample images (folder)
    print(args.sample_folder_path)
    img_resize = load_image_folder(args.sample_folder_path, args.img_size, args.batch_size)
    '''# Sample (one image)
    print(args.sample_image_path)
    img_resize, img_raw = load_image(args.sample_image_path, args.img_size)'''
            
    print('inference image size:', img_resize.shape)

    # Build TensorRT engine
    #build_engine(args.onnx_model_path, args.min_batch_size, args.max_batch_size, args.img_size, args.output_shape, args.fp16_mode, args.dynamic_axes, args.tensorrt_engine_path)
    build_engine(args.onnx_model_path, args.fp16_mode, args.tensorrt_engine_path)

    # Read the engine from the file and deserialize
    with open(args.tensorrt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime: 
        engine = runtime.deserialize_cuda_engine(f.read())    
    context = engine.create_execution_context()
    
    # Allocating memory
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    # TensorRT inference
    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    inputs[0].host = img_resize[:5]
    
    trt_start_time = time.time()
    trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=args.batch_size)
    trt_end_time = time.time()
    
    # ONNX inference
    onnx_model = onnx.load(args.onnx_model_path)
    sess = rt.InferenceSession(args.onnx_model_path)
    
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    
    sess_input = sess.get_inputs()[0].name
    sess_output = sess.get_outputs()[0].name
    
    onnx_start_time = time.time()
    onnx_result = sess.run([sess_output], {sess_input: img_resize.astype(np.float32)})[0]
    onnx_end_time = time.time()
    
    ## Comparision output of TensorRT and output of onnx model

    # Time Efficiency & output
    print('--onnx--')
    print(onnx_result.shape)
    print(np.argmax(onnx_result, axis=1))
    print('time: ', onnx_end_time - onnx_start_time)
    
    print('--tensorrt--')
    trt_outputs = np.array(trt_outputs).reshape(args.batch_size, -1)
    print(trt_outputs.shape)
    print(np.argmax(trt_outputs, axis=1))
    print('time: ', trt_end_time - trt_start_time)
