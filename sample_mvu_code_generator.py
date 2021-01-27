import argparse
from OnnxParser import OnnxParser
from Generator import Generator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--onnx_model', help='input onnx model', required=True)
    parser.add_argument('--aprec', help='Activation precision', required=False, default=8, type=int)
    parser.add_argument('--wprec', help='Weight precision', required=False, default=8, type=int)
    parser.add_argument('--oprec', help='Output precision', required=False, default=8, type=int)
    parser.add_argument('--input_shape', help='input shape for ',  nargs='*', required=False, default=[3,32,32], type=int)
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    args = parse_args()
    model = OnnxParser(args['onnx_model'])

    # model.print_onnx_graph()
    # model.print_onnx_model()
    precision = [args['aprec'], args['wprec'], args['oprec']]
    # # import ipdb as pdb; pdb.set_trace()
    if len(args['input_shape'])>3:
        print("Expecting an input array of shape: [channels, height, lenghth]")
        import sys
        sys.exit()
    input_shape = args['input_shape']
    generator = Generator(model, precision, input_shape)
    generator.generate_mvu_configs()
    generator.export_weigths()

