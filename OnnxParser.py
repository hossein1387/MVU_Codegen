import onnx 
from texttable import Texttable
import numpy as np
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

class OnnxParser():
    """docstring for OnnxParser"""
    def __init__(self, onnx_mdoel_path):
        super(OnnxParser, self).__init__()
        self.onnx_model = onnx.load(onnx_mdoel_path)
        self.model_name = onnx_mdoel_path.split("/")[-1].split(".")[0]
        self.layers = []
        # self.matmul_layers = []
        self.parsed_graph = []
        self.parse()

    # For now we only parse convolution layers
    def parse(self):
        dims = self.get_conv_dims()
        conv_num = 0
        layers = []
        graph = []
        for node in self.onnx_model.graph.node:
            if node.op_type.lower() == "conv":
                # import ipdb as pdb; pdb.set_trace()
                attribs = self.get_onnx_conv_attrib(node, dims)
            elif node.op_type.lower() == "gemm" or node.op_type.lower() == "matmul":
                attribs = self.get_onnx_matmul_attrib(node, dims)
            self.layers.append(attribs)
            print(attribs['layer_name'])
            graph.append(self.get_onnx_graph_attrib(node))

        self.parsed_graph = graph

    def get_node_weight(self, node):
        for initializer in self.onnx_model.graph.initializer:
            if node.input[1] == initializer.name:
                dtype = TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]
                np.frombuffer(initializer.raw_data, dtype=dtype)
                np.frombuffer(initializer.raw_data, dtype=dtype)
                weight = np.frombuffer(initializer.raw_data, dtype=dtype).reshape(*initializer.dims) 
                return weight, initializer.name
        print("==> Warning: could not find weight param for {}".format(node.input['1']))
        return None

    def get_onnx_conv_attrib(self, conv_node, dims):
        # import ipdb as pdb; pdb.set_trace()
        attribs = {}
        if conv_node.name == "":
            attribs['layer_name'] = "Conv_{}".format(len(self.layers))
        else:
            attribs['layer_name'] = conv_node.name
        attribs['layer_type'] = "conv"
        attribs['in_channels'] = dims[conv_node.input[1]][1]
        attribs['out_channels'] = dims[conv_node.input[1]][0]
        attribs['dilation'] = conv_node.attribute[0].ints
        attribs['groups'] = conv_node.attribute[1].i
        attribs['kernel_size'] = conv_node.attribute[2].ints
        attribs['padding'] = conv_node.attribute[3].ints
        attribs['stride'] = conv_node.attribute[4].ints
        attribs['weight'], attribs['name'] = self.get_node_weight(conv_node)
        return attribs

    def get_onnx_matmul_attrib(self, matmul_node, dims):
        # import ipdb as pdb; pdb.set_trace()
        attribs = {}
        if matmul_node.name == "":
            attribs['layer_name'] = "Conv_{}".format(len(self.layers))
        else:
            attribs['layer_name'] = matmul_node.name
        attribs['layer_type'] = "matmul"
        attribs['in_channels'] = dims[matmul_node.input[1]][1]
        attribs['out_channels'] = dims[matmul_node.input[1]][0]
        attribs['dilation'] = "N/A"
        attribs['groups'] = "N/A"
        attribs['padding'] = "N/A"
        attribs['stride'] = "N/A"
        attribs['weight'], attribs['name'] = self.get_node_weight(matmul_node)
        attribs['kernel_size'] = attribs['weight'].shape
        return attribs


    def get_onnx_graph_attrib(self, node):
        attribs = {}
        all_int = True
        all_inps = []
        for inp in node.input:
            all_int *= inp.isdecimal()
            if inp.isdecimal():
                all_inps.append(inp)
        attribs['input_node'] = all_inps if all_int else [node.input[0]]
        attribs['output_node'] = node.output
        attribs['node_type'] = node.op_type
        return attribs

    def get_conv_dims(self):
        dims = {}
        for val in self.onnx_model.graph.initializer:
            dims[val.name] = val.dims
        return dims

    def print_onnx_model(self):
        t = Texttable(max_width=180)
        t.add_row(['lay_num', 'type', 'in_channels','out_channels','kernel_size','stride','padding','dilation','groups'])
        dims = self.get_conv_dims()
        lay_num = 0
        for layer in self.layers:
            t.add_row([lay_num,
                       layer['layer_type'],
                       layer['in_channels'],
                       layer['out_channels'],
                       layer['kernel_size'],
                       layer['stride'],
                       layer['padding'],
                       layer['dilation'],
                       layer['groups']])
            lay_num += 1
        print("Onnx Model Configurations:")
        print(t.draw())

    def print_onnx_graph(self):
        t = Texttable(max_width=180)
        t.add_row(['lay_num', 'input_node', 'output_node', 'node_type'])
        dims = self.get_conv_dims()
        lay_num = 0
        for layer in self.parsed_graph:
            t.add_row([lay_num,
                       layer['input_node'],
                       layer['output_node'],
                       layer['node_type']])
            lay_num += 1
        print("Onnx Model Graph:")
        print(t.draw())
