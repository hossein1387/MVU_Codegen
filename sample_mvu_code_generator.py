import argparse
from OnnxParser import OnnxParser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--onnx_model', help='input onnx model', required=True)
    parser.add_argument('--aprec', help='Activation precision', required=False, default=8)
    parser.add_argument('--wprec', help='Weight precision', required=False, default=8)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = parse_args()
    parsed_model = OnnxParser(args['onnx_model'])
    parsed_model.print_onnx_model()
