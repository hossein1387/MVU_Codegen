import onnx 
from texttable import Texttable
import numpy as np
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx import shape_inference

def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    return add_const_value_infos_to_graph(model.graph)

class OnnxParser():
    """docstring for OnnxParser"""
    def __init__(self, onnx_mdoel_path, input_shape):
        super(OnnxParser, self).__init__()
        self.onnx_model = onnx.load(onnx_mdoel_path)
        self.model_name = onnx_mdoel_path.split("/")[-1].split(".")[0]
        self.layers = []
        # self.matmul_layers = []
        self.input_shape = input_shape
        self.parsed_graph = []
        self.node_shape_info = {}
        self.infer_shapes()
        self.parse()

    def get_dim(self, node: onnx.onnx_ml_pb2.ValueInfoProto):
        dims = []
        for dim in node.type.tensor_type.shape.dim:
            dims.append(dim.dim_value)
        return dims

    def infer_shapes(self):
        # import ipdb as pdb; pdb.set_trace()
        add_value_info_for_constants(self.onnx_model)
        inferred_model = shape_inference.infer_shapes(self.onnx_model)
        for value in inferred_model.graph.value_info:
            self.node_shape_info[value.name] = self.get_dim(value)
        # finally, lets add shape for input and output nodes
        self.node_shape_info[self.onnx_model.graph.input[0].name] = self.input_shape
        self.node_shape_info[self.onnx_model.graph.output[0].name] = self.get_dim(self.onnx_model.graph.output[0])


    # For now we only parse convolution layers
    def parse(self):
        dims = self.get_conv_dims()
        conv_num = 0
        layers = []
        graph = []
        for node in self.onnx_model.graph.node:
            # import ipdb as pdb; pdb.set_trace()
            if node.op_type.lower() == "conv":
                # import ipdb as pdb; pdb.set_trace()
                attribs = self.get_onnx_conv_attrib(node, dims)
                attribs['input_shape'] = self.node_shape_info[node.input[0]]
                attribs['output_shape'] = self.node_shape_info[node.output[0]]
            elif node.op_type.lower() == "gemm" or node.op_type.lower() == "matmul":
                attribs = self.get_onnx_matmul_attrib(node, dims)
                attribs['input_shape'] = self.node_shape_info[node.input[0]]
                attribs['output_shape'] = self.node_shape_info[node.output[0]]
            else:
                attribs['input_shape'] = self.node_shape_info[node.input[0]]
                attribs['output_shape'] = self.node_shape_info[node.output[0]]

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
        attribs['weight'], attribs['node_name'] = self.get_node_weight(conv_node)
        return attribs

    def get_onnx_matmul_attrib(self, matmul_node, dims):
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
        attribs['weight'], attribs['node_name'] = self.get_node_weight(matmul_node)
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
