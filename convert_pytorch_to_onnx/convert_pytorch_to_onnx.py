import argparse
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from collections import OrderedDict
from PIL import Image
import cv2

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

def parse_args():
    parser = argparse.ArgumentParser(description='Convert Pytorch models to ONNX')
    parser.add_argument('--device', help='cuda or not',
        default='cuda:0')
    parser.add_argument('--img_size', help='image size',
        default=[1, 3, 224, 224])
    parser.add_argument('--batch_size', type=int, help='onnx sample batch size',
        default=1)
    parser.add_argument('--output_path',  help='experted onnx path',
        default='./onnx_output.onnx')
    parser.add_argument('--sample_image_path', help='sample image path',
        default='./sample.jpg')
    parser.add_argument('--sample_folder_path', help='sample image folder path',
        default='./imagenet-mini/train/')
    parser.add_argument('--dynamic_axes', help='dynamic batch input or output',
        default='False')
    parser.add_argument('--keep_initializers_as_inputs', help='If True, all the initializers (typically corresponding to parameters) in the exported graph will also be added as inputs to the graph. If False, then initializers are not added as inputs to the graph, and only the non-parameter inputs are added as inputs.',
        default='True')
    parser.add_argument('--export_params', help='If specified, all parameters will be exported. Set this to False if you want to export an untrained model.',
        default='True')
    parser.add_argument('--opset_version', type=int, help='opset version',
        default=11)
    args = string_to_bool(parser.parse_args())

    return args

def string_to_bool(args):

    if args.dynamic_axes.lower() in ('true'): args.dynamic_axes = True
    else: args.dynamic_axes = False

    if args.keep_initializers_as_inputs.lower() in ('true'): args.keep_initializers_as_inputs = True
    else: args.keep_initializers_as_inputs = False

    if args.export_params.lower() in ('true'): args.export_params = True
    else: args.export_params = False

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

def model_train():
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam()

if __name__ == '__main__':
    args = parse_args()

    # Load pretrained model
    resnet18 = models.resnet18(pretrained=True)
    
    '''fc = nn.Sequential(OrderedDict([
      ('fc1', nn.Linear(512,1000)),
      ('output',nn.Softmax(dim=1))
    ]))
    resnet18.fc = fc'''
    
    print(resnet18)
    
    resnet18.cpu().eval()

    # Sample images (folder)
    print(args.sample_folder_path)
    img_resize = load_image_folder(args.sample_folder_path, args.img_size, args.batch_size)
    '''# Sample (one image)
    print(args.sample_image_path)
    img_resize, img_raw = load_image(args.sample_image_path, args.img_size)'''
    
    sample_input = torch.randn(args.batch_size, args.img_size[1], args.img_size[2], args.img_size[3])
    print('inference image size:', img_resize.shape, 'sample input size:', sample_input.shape)
    
    if args.dynamic_axes:
        # Dynamic input
        dynamic_axes = {'input' : {0 : 'batch_size'}, 
                                'output' : {0 : 'batch_size'}}
                                
        # Export onnx
        torch.onnx.export(
        resnet18,
        sample_input,
        args.output_path,
        export_params=args.export_params,
        keep_initializers_as_inputs=args.keep_initializers_as_inputs,
        opset_version=args.opset_version,
        input_names=['input'], # input vect 이름 설정
        output_names=['output'], # output vect 이름 설정
        dynamic_axes=dynamic_axes, # dynamic input 지정
        verbose=False)
    else:
        # Export onnx
        torch.onnx.export(
        resnet18,
        sample_input,
        args.output_path,
        export_params=args.export_params,
        keep_initializers_as_inputs=args.keep_initializers_as_inputs,
        opset_version=args.opset_version,
        verbose=False)        

    # Load the ONNX model
    onnx_model = onnx.load(args.output_path)

    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)

    # Print a human readable representation of the graph
    with open('OnnxShape.txt','w') as f:
            f.write(f"{onnx.helper.printable_graph(onnx_model.graph)}")

    # ONNX inference using onnxruntime
    sess = rt.InferenceSession(args.output_path)
    sess_input = sess.get_inputs()[0].name
    sess_output = sess.get_outputs()[0].name
    pred = sess.run([sess_output], {sess_input: img_resize.astype(np.float32)})[0]
    #print(pred.shape)
    
    ## Comparision output of onnx and output of Pytorch model
    # Pytorch results
    img_resize_torch = torch.Tensor(img_resize)
    pytorch_result = resnet18(img_resize_torch).detach().numpy()

    # ONNX results
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    sess = rt.InferenceSession(args.output_path)
    onnx_result = sess.run(None,
                            {net_feed_input[0]: img_resize
                            })[0]
    
    print('--pytorch--')
    print(pytorch_result.shape) # (batch_size, 1000)
    print(pytorch_result)
    print(np.argmax(pytorch_result, axis=1))

    print('--onnx--')
    print(onnx_result.shape)
    print(onnx_result)
    print(np.argmax(onnx_result, axis=1))

    # Comparision
    assert np.allclose(
        pytorch_result, onnx_result,
        atol=1.e-5), 'The outputs are different (Pytorch and ONNX)'
    print('The numerical values are same (Pytorch and ONNX)')
