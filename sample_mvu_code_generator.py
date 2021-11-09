import logging
import argparse
from OnnxParser import OnnxParser
from Generator import Generator
import utils
import json
  
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--onnx_model', help='input onnx model', required=True)
    parser.add_argument('--input_shape', help='input shape for ',  nargs='*', required=False, default=[1, 3,32,32], type=int)
    parser.add_argument('--mvu_cfg', help='MVU configuration file ', required=False)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = parse_args()
    model_path = args['onnx_model']
    # import ipdb as pdb; pdb.set_trace()
    with open(args['mvu_cfg']) as json_file:
        mvu_cfg = json.load(json_file)

    input_shape = args['input_shape']
    model = OnnxParser(model_path, input_shape)

    # model.print_onnx_graph()
    # model.print_onnx_model()
    if len(args['input_shape'])>4:
        print("Expecting an input array of shape: [in_channels, out_channels, height, lenghth]")
        import sys
        sys.exit()
    # import ipdb as pdb; pdb.set_trace()
    generator = Generator(model, input_shape, mvu_cfg)
    generator.generate_mvu_configs()
    generator.export_weigths()
    precision = [mvu_cfg['aprec'], mvu_cfg['wprec'], mvu_cfg['oprec']]
    utils.gen_test_vecs(model_path, precision, input_shape)
