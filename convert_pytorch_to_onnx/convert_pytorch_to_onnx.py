import argparse
import os
import numpy as np
from skimage import io
from skimage.transform import resize

# Torch
import torch
import torchvision.models as models
from torch.utils.model_zoo import load_url as load_state_dict_from_url

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
    parser.add_argument('--input_shape', help='input size',
        default=[1, 3, 224, 192])
    parser.add_argument('--load_file', help='load model using file path or url',
        default='True')
    parser.add_argument('--checkpoint_file_path', help='checkpoint file path',
        default='./resnet18-f37072fd.pth')
    parser.add_argument('--checkpoint_download_url', help='checkpoint download url',
        default='https://download.pytorch.org/models/resnet18-f37072fd.pth')
    parser.add_argument('--output_path',  help='experted onnx path',
        default='./onnx_output.onnx')
    parser.add_argument('--sample_image_path', help='sample image path',
        default='./sample.jpg')
    parser.add_argument('--keep_initializers_as_inputs', help='If True, all the initializers (typically corresponding to parameters) in the exported graph will also be added as inputs to the graph. If False, then initializers are not added as inputs to the graph, and only the non-parameter inputs are added as inputs.',
        default='True')
    parser.add_argument('--export_params', help='If specified, all parameters will be exported. Set this to False if you want to export an untrained model.',
        default='True')
    parser.add_argument('--opset_version', type=int, help='opset version',
        default=9)
    #parser.add_argument('--verify',action='store_true',help='verify the onnx model')
    #parser.add_argument('--shape-inference', action='store_true', help='graph value of inferenced_onnx')
    #parser.add_argument('--run', action='store_true', help='real-data inference')
    args = string_to_bool(parser.parse_args())

    return args

def string_to_bool(args):
    
    if args.load_file.lower() in ('true'): args.load_file = True
    else: args.load_file = False

    if args.keep_initializers_as_inputs.lower() in ('true'): args.keep_initializers_as_inputs = True
    else: args.keep_initializers_as_inputs = False

    if args.export_params.lower() in ('true'): args.export_params = True
    else: args.export_params = False

    return args

def load_image(img_path, size):
    img_raw = io.imread(img_path)
    #print('img_raw.shape:', img_raw.shape)
    img_raw = np.rollaxis(img_raw, 2, 0)
    img_resize = resize(img_raw / 255, size, anti_aliasing=True)
    #img_resize = img_resize[np.newaxis, :, :, :]
    img_resize = img_resize.astype(np.float32)
    #print('img_resize.shape:', img_resize.shape)
    return img_resize, img_raw

if __name__ == '__main__':
    args = parse_args()

    # Load pretrained model
    if args.load_file:
        print('load pth file...')
        checkpoint = torch.load(args.checkpoint_file_path, map_location=args.device)
    else:
        print('download file...')
        checkpoint = load_state_dict_from_url(args.checkpoint_download_url, map_location=args.device)
    #print(checkpoint)
    resnet18 = models.resnet18()
    #print(resnet18)
    resnet18.load_state_dict(checkpoint)
    resnet18.cpu().eval()

    # Sample
    sample_input = torch.randn(args.input_shape)
    img_resize, img_raw = load_image(args.sample_image_path, args.input_shape)

    # Export onnx
    torch.onnx.export(
    resnet18,
    sample_input,
    args.output_path,
    export_params=args.export_params,
    keep_initializers_as_inputs=args.keep_initializers_as_inputs,
    opset_version=args.opset_version,
    verbose=True)

    # Load the ONNX model
    onnx_model = onnx.load(args.output_path)

    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)

    # Print a human readable representation of the graph
    with open('OnnxShape_'+os.path.basename(args.checkpoint_file_path)+'.txt','w') as f:
            f.write(f"{onnx.helper.printable_graph(onnx_model.graph)}")

    # ONNX inference using onnxruntime
    sess = rt.InferenceSession(args.output_path)
    sess_input = sess.get_inputs()[0].name
    sess_output = sess.get_outputs()[0].name
    pred = sess.run([sess_output], {sess_input: img_resize.astype(np.float32)})[0]
    print(pred.shape)

    ## Comparision output of onnx with Pytorch
    # Pytorch results
    pytorch_result = resnet18(sample_input).detach().numpy()
    print(pytorch_result[0].shape)

    # ONNX results
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    sess = rt.InferenceSession(args.output_path)
    onnx_result = sess.run(None,
                            {net_feed_input[0]: sample_input.detach().numpy()
                            })[0]
    print(onnx_result[0].shape)

    # Comparision
    assert np.allclose(
        pytorch_result, onnx_result,
        atol=1.e-5), 'The outputs are different (Pytorch and ONNX)'
    print('The numerical values are same (Pytorch and ONNX)')
