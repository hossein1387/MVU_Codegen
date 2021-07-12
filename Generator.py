from texttable import Texttable
from math import ceil
import sys
import numpy as np
from utils import export_tensor
import copy

class Generator():
    """docstring for Generator"""
    def __init__(self, model, prec, input_shape):
        super(Generator, self).__init__()
        # expecting to receive a OnnxModel parsed object
        self.model = model
        self.prec = prec
        self.input_shape = copy.copy(input_shape)

    def check_model_is_valid(self):
        # check if there are residual connections
        for layer in self.model.parsed_graph:
            # import ipdb as pdb; pdb.set_trace()
            if len(layer['input_node']) > 1 and (layer['node_type']!="Reshape"):
                # import ipdb as pdb; pdb.set_trace()
                print(" ==> Models with residual connections are not supported <==")
                print(" ==>                      :(                            <==")
                sys.exit()

    # iprec: Input data precision
    # wprec: Weight precision
    # oprec: Output precision
    # iW    : Input width
    # iH    : Input height
    # iC    : Input channel blocks
    # fW   : Filter kernel width
    # fH   : Filter kernel height
    # fC   : Number of filter set blocks (i.e. number of output channel blocks)
    # sW   : Filter horizontal (width) stride
    # Pl   : Zero-padding in on the left in the width dimension
    # Pr   : Zero-padding in on the right in the width dimension
    # Pt   : Zero-padding in on the top in the height dimension
    # Pb   : Zero-padding in on the bottom in the height dimension

    def get_mvu_param(self, prec, iShape, fShape, stride, layerType):
        iprec,wprec,oprec = prec
        iC, iH, iW = iShape
        fC, fH, fW = fShape
        sW = stride 
        ilength = [0,0,0,0,0]
        ijump   = [0,0,0,0,0]
        wlength = [0,0,0,0,0]
        wjump   = [0,0,0,0,0]

        if layerType == "conv":
            ilength[4] = 0
            ilength[3] = iC*fW-1
            ilength[2] = fH-1
            ilength[1] = iprec*wprec*fC-1

            ijump[4] = 0
            ijump[3] = iprec
            ijump[2] = iprec*(iC*(iW-fW) + 1)
            ijump[1] = -iprec*(iC*(fH-1)*iW + fW*iC - 1)
            ijump[0] = -iprec*(iC*(fH-1)*iW + (fW-sW-1)*iC + 1)

            wlength[4] = 0
            wlength[3] = iC*fW*fH-1
            wlength[2] = iprec*wprec-1
            wlength[1] = fC-1

            wjump[4] = 0
            wjump[3] = wprec
            wjump[2] = -wprec*(iC*fW*fH-1)
            wjump[1] = wprec
            wjump[0] = -wprec*(iC*fW*fH*fC-1)
            countdown = (iC * fW) * (fH) * (iprec * wprec) * (fC) * ((iW-fW+1)/sW)
            
        elif layerType == "matmul":
            m_h = ceil(iH/64)
            m_w = ceil(iW/64)
            ilength[4] = 0
            ilength[3] = 0
            ilength[2] = 0
            ilength[1] = m_h-1

            ijump[4] = 0
            ijump[3] = 0
            ijump[2] = 0
            ijump[1] = iprec
            ijump[0] = -iprec*(m_w-1)

            wlength[4] = 0
            wlength[3] = 0
            wlength[2] = m_w-1
            wlength[1] = wprec*iprec-1

            wjump[4] = 0
            wjump[3] = 0
            wjump[2] = wprec
            wjump[1] = -wprec*(m_w-1)
            wjump[0] = wprec

            countdown = m_w * m_h * iprec * wprec;

        return [ilength, ijump, wlength, wjump, countdown] 

    def infer_activation_shape(self, input, kernel, padding, stride, layerTpye):
        oC, oH, oW = [0, 0, 0]
        if layerTpye == "conv":
            iC, iH, iW = input
            fC, fH, fW = kernel
            oH=int((iH-fH+2*padding)/stride)+1
            oW=int((iW-fW+2*padding)/stride)+1
            oC=fC
        elif layerTpye == "matmul":
            pass
        return [oC, oH, oW]

    def generate_mvu_configs(self):
        self.check_model_is_valid()
        t = Texttable(max_width=160)
        t.add_row(['iShape', 'fShape', 'ilength', 'ijump', 'wlength', 'wjump', 'countdown', 'total layer countdown'])
        # import ipdb as pdb; pdb.set_trace()
        input_shape = self.input_shape
        input_shape[0] = ceil(input_shape[0]/64)
        total_cycles = 0
        for layer in self.model.layers:
            layer_type = layer['layer_type']
            iShape = input_shape
            fShape = [ceil(layer['out_channels']/64), layer['kernel_size'][0], layer['kernel_size'][1]]
            stride = layer['stride'][0]
            padding = layer['padding'][0]
            prec = self.prec
            # print("{} * {}".format(iShape, fShape))
            ilength, ijump, wlength, wjump, countdown = self.get_mvu_param(prec, iShape, fShape, stride, layer_type)
            if layer_type == "conv":
                total_layer_countdown = countdown * ceil((input_shape[2]+layer['padding'][0]+layer['padding'][1]) / layer['stride'][1])
            elif layer_type == "matmul":
                total_layer_countdown = countdown
            # import ipdb as pdb; pdb.set_trace()
            t.add_row([iShape, fShape, ilength, ijump, wlength, wjump, countdown, total_layer_countdown])
            input_shape = self.infer_activation_shape(input_shape, fShape, padding, stride, layer_type)
            total_cycles += total_layer_countdown
        print("\nGenerated MVU configuration:")
        print(t.draw())
        print("Total countdown: {}".format(int(total_cycles)))


    def print_mvu_param(self):
        pass

    def int2bit(self, val, precision):
        val_str = '{0:032b}'.format(val)
        return list(val_str[::-1][0:precision][::-1])

    def export_weigths(self):
        weight_dict = self.__process_weigths()
        export_tensor(weight_dict)

    def __export_weigths(self, dict):
        for key, vals in dict.items():
            file_name = "{}.hex".format(key)
            with open(file_name, "w") as f:
                for val in vals:
                    f.write("{}\n".format(val))
            f.close()
            print("Exporting {} to {}".format(key, file_name))

    def __process_weigths(self):
        iprec,wprec,oprec = self.prec
        # expecting input shapes is:
        # [input_channels, output_channels, width, height]
        weight_ram = {}
        for layer in self.model.layers:
            layer_weights = []
            layer_type = layer['layer_type']
            # first we need to transpose the weight tensor into channel first format
            # import ipdb as pdb; pdb.set_trace()
            if layer_type == "conv":
                weights = layer['weight'].transpose(3,2,1,0)
            elif layer_type == "matmul":
                weights = layer['weight']
            # The accelerator only works with integer values
            flatten_weights = [int(val) for val in weights.flatten()]
            # print(flatten_weights)
            weight_block = np.zeros([4096, wprec],dtype=str)
            cnt = 0
            processed = False
            # Now we need to transpose the weight tensor into MVU format.
            for idx, val in enumerate(flatten_weights):
                if cnt >= 4096 or idx==(len(flatten_weights)-1):
                    cnt = 0
                    for weight in weight_block.transpose(1,0):
                        # import ipdb as pdb; pdb.set_trace()
                        val_str = "".join(weight)
                        val_str = val_str.zfill(4096)[::-1]
                        layer_weights.append(val_str)
                        processed = True
                weight_block[cnt] = self.int2bit(val, wprec)
                cnt += 1
            # import ipdb as pdb; pdb.set_trace()
            if not processed:
                # import ipdb as pdb; pdb.set_trace()
                for weight in weight_block.transpose(1,0):
                    val_str = "".join(weight)
                    #val_str = val_str[::-1].zfill(4096)
                    layer_weights.append(val_str)
            weight_ram[layer['name']] = layer_weights
        return weight_ram
        
