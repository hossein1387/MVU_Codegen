from texttable import Texttable
from math import ceil
import sys
import numpy as np
from utils import export_tensor
import copy
from collections import OrderedDict

class MVUConfig():
    def __init__(self, prec, meminfo, quantIdx, ilength, ijump, wlength, wjump, countdown, olength):
        super(MVUConfig, self).__init__()
        self.prec = prec
        self.meminfo = meminfo
        self.quantIdx = quantIdx
        self.ilength = ilength
        self.ijump = ijump
        self.wlength = wlength
        self.wjump = wjump
        self.countdown = countdown
        self.olength = olength

class Generator():
    """docstring for Generator"""
    def __init__(self, model, prec, input_shape, meminfo, quantIdx=2, temp_riscv_code_file="template.S"):
        super(Generator, self).__init__()
        # expecting to receive a OnnxModel parsed object
        self.model = model
        self.prec = prec
        self.input_shape = copy.copy(input_shape)
        self.temp_riscv_code_file = temp_riscv_code_file
        self.meminfo = meminfo
        self.quantIdx = quantIdx
        self.loop_cnt = 0
        self.mvu_res_output_base_addr = 1024
        self.mvu_res_input_base_addr = 0
        self.mvu_res_weight_base_addr = 0

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
        # import ipdb as pdb; pdb.set_trace()
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
            m_h = ceil(fH/64)
            m_w = ceil(fW/64)
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

    def _get_riscv_csr_code(self, mvuConfig, layer_type, layer_name):
        iprec,wprec,oprec = mvuConfig.prec
        imem, wmem, omem  = mvuConfig.meminfo

        def gen_csr_instr(csr, val):
            # risc-v immediate CSRs are small and can take values from 0-31. 
            # here we check if the absolute csr value is larger than 31. If so,
            # use regular immediate instructions to load the value into general
            # registers and then load them into csr with csrw instruction. If not,
            # then that can be handeled with a simple csrwi instruciton.
            val = int(val)
            if abs(val)>((2**5)-1):
                _temp_str = ""
                _temp_str += "\taddi t1, {}\n".format(val)
                _temp_str += "\tcsrw {} , t1\n".format(csr)
                return _temp_str
            else:
                if val >= 0:
                    return "\tcsrwi {0}, {1}\n".format(csr, val)
                else:
                    return "\tcsrwi {0}, {1}\n".format(csr, slice_val(val, 5))

        # 32 bit negative numbers are represented as 5 bit integer values,
        # it is like slicing the value in binary with 5 bits.
        def slice_val(val, bits):
            def int2bin(integer, digits):
                if integer >= 0:
                    return bin(integer)[2:].zfill(digits)
                else:
                    return bin(2**digits + integer)[2:]
            str_bin = int2bin(val, bits)
            return int(str_bin[0:bits], 2)
        code_str = ""
        if layer_type == "matmul":
            # Matmul compute structure code
            code_str += "{}:\n".format(layer_name)
            code_str += "\taddi sp, sp, -4\n"
            code_str += "\tsw ra, 4(sp)\n"
            code_str += "\tjal {}_init\n".format(layer_name)
            code_str += "\tjal {}_loop\n".format(layer_name)
            code_str += "\tlw ra, 4(sp)\n"
            code_str += "\taddi sp, sp, 4\n"
            code_str += "\tret\n"

            # Matmul initialization code
            code_str += "{}_init:\n".format(layer_name)
            code_str += "\taddi sp, sp, -4\n"
            code_str += "\tsw ra, 4(sp)\n"
            code_str += "\taddi  t1, x0, 0\n"
            code_str += "\taddi  t2, x0, {}\n".format(wprec)
            code_str += "\tadd   t1, t1, t2\n"
            code_str += "\taddi  t2, x0, {}\n".format(iprec)
            code_str += "\tslli  t3, t2, 6\n"
            code_str += "\tadd   t1, t1, t3\n"
            code_str += "\taddi  t2, x0, {}\n".format(oprec)
            code_str += "\tslli  t3, t2, 12\n"
            code_str += "\tadd   t1, t1, t3\n"
            code_str += "\tcsrw  mvuprecision,  t1\n"

            code_str += gen_csr_instr("mvuwjump_0", mvuConfig.wjump[0])
            code_str += gen_csr_instr("mvuwjump_1", mvuConfig.wjump[1])
            code_str += gen_csr_instr("mvuwjump_2", mvuConfig.wjump[2])
            code_str += gen_csr_instr("mvuwjump_3", mvuConfig.wjump[3])
            
            code_str += gen_csr_instr("mvuijump_0", mvuConfig.ijump[0])
            code_str += gen_csr_instr("mvuijump_1", mvuConfig.ijump[1])
            code_str += gen_csr_instr("mvuijump_2", mvuConfig.ijump[2])
            code_str += gen_csr_instr("mvuijump_3", mvuConfig.ijump[3])

            code_str += "\tcsrwi {0}, {1}\n".format("mvusjump_0", 0)
            code_str += "\tcsrwi {0}, {1}\n".format("mvusjump_1", 0)

            code_str += "\tcsrwi {0}, {1}\n".format("mvubjump_0", 0)
            code_str += "\tcsrwi {0}, {1}\n".format("mvubjump_1", 0)

            code_str += "\tcsrwi {0}, {1}\n".format("mvuojump_0", 0)
            code_str += "\tcsrwi {0}, {1}\n".format("mvuojump_1", 0)
            code_str += "\tcsrwi {0}, {1}\n".format("mvuojump_2", 0)
            code_str += "\tcsrwi {0}, {1}\n".format("mvuojump_3", 0)
            code_str += "\tcsrwi {0}, {1}\n".format("mvuojump_4", 0)

            code_str += gen_csr_instr("mvuwlength_1", mvuConfig.wlength[1])
            code_str += gen_csr_instr("mvuwlength_2", mvuConfig.wlength[2])
            code_str += gen_csr_instr("mvuwlength_3", mvuConfig.wlength[3])
            code_str += gen_csr_instr("mvuwlength_4", mvuConfig.wlength[4])

            code_str += gen_csr_instr("mvuilength_1", mvuConfig.ilength[1])
            code_str += gen_csr_instr("mvuilength_2", mvuConfig.ilength[2])
            code_str += gen_csr_instr("mvuilength_3", mvuConfig.ilength[3])
            code_str += gen_csr_instr("mvuilength_4", mvuConfig.ilength[4])

            code_str += gen_csr_instr("mvuolength_1", mvuConfig.ilength[1])
            code_str += gen_csr_instr("mvuolength_2", mvuConfig.ilength[2])
            code_str += gen_csr_instr("mvuolength_3", mvuConfig.ilength[3])
            code_str += gen_csr_instr("mvuolength_4", mvuConfig.ilength[4])

            code_str += "\tlw ra, 4(sp)\n"
            code_str += "\taddi sp, sp, 4\n"
            code_str += "\tret\n"
            
            # Matmul loop code
            # loop init:
            code_str += "{}_loop:\n".format(layer_name)
            code_str += "\taddi sp, sp, -4\n"
            code_str += "\tsw ra, 4(sp)\n"
            code_str += "\taddi s0, x0, 64\n"
            code_str += "\tli s1, {}\n".format(self.mvu_res_output_base_addr)
            code_str += "\tli s2, {}\n".format(self.mvu_res_weight_base_addr)
            code_str += "\tli s3, {}\n".format(self.mvu_res_input_base_addr)
            code_str += "\taddi s4, x0, 1\n"
            code_str += "\tslli s4, s4, 30\n"
            code_str += "\taddi s4, s4, {}\n".format(int(mvuConfig.countdown))
            # loop anchor:
            code_str += "loop_{}:\n".format(self.loop_cnt)
            code_str += gen_csr_instr("mvuquant", oprec-1)
            code_str += "\tcsrw mvuwbaseptr, s2\n"
            code_str += "\tcsrw mvuibaseptr, s3\n"
            code_str += "\tcsrw mvuobaseptr, s1\n"
            code_str += "\tcsrw mvucommand, s4\n"
            code_str += "\tjal wait_for_mvu_irq\n"
            code_str += "\taddi s0,s0, -1\n"
            code_str += "\taddi s1, s1, {}\n".format(oprec)
            code_str += "\taddi s3, s3, {}\n".format(iprec)
            code_str += "\tbne s0, x0, loop_{}\n".format(self.loop_cnt)
            code_str += "\tlw ra, 4(sp)\n"
            code_str += "\taddi sp, sp, 4\n"
            code_str += "\tret\n"

            self.loop_cnt += 1

        return code_str

    def __gen__riscv_code(self, model, _func_dict):
        temp_str = ""
        with open(self.temp_riscv_code_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "--> FUNCCALL <--" in line:
                    for key, value in _func_dict.items():
                        temp_str += "\tjal {}\n".format(key)
                elif "--> HERE <--" in line:
                    for key, value in _func_dict.items():
                        temp_str += "{}\n".format(value)
                else:
                    temp_str += line
            f.close()
        output_file = "{}.S".format(model.model_name)
        with open(output_file, 'w') as f:
            f.write(temp_str)
            f.close()
        print("Generated kernel code is written to {}".format(output_file))

    def generate_mvu_configs(self):
        self.check_model_is_valid()
        t = Texttable(max_width=160)
        t.add_row(['iShape', 'fShape', 'ilength', 'ijump', 'wlength', 'wjump', 'countdown', 'total layer countdown'])
        # import ipdb as pdb; pdb.set_trace()
        input_shape = self.input_shape
        input_shape[0] = ceil(input_shape[0]/64)
        total_cycles = 0
        _func_dict = OrderedDict()
        for layer in self.model.layers:
            _code_str = ""
            layer_type = layer['layer_type']
            layer_name = layer['layer_name']
            iShape = input_shape
            fShape = [ceil(layer['out_channels']/64), layer['kernel_size'][0], layer['kernel_size'][1]]
            stride = layer['stride'][0]
            padding = layer['padding'][0]
            prec = self.prec
            # print("{} * {}".format(iShape, fShape))
            # import ipdb as pdb; pdb.set_trace()
            ilength, ijump, wlength, wjump, countdown = self.get_mvu_param(prec, iShape, fShape, stride, layer_type)
            olength = [0,1,0,0,0]
            mvuConfig = MVUConfig(self.prec, self.meminfo, self.quantIdx, ilength, ijump, wlength, wjump, countdown, olength)
            _code_str += self._get_riscv_csr_code(mvuConfig, layer_type, layer_name)
            _func_dict[layer_name] = _code_str
            if layer_type == "conv":
                total_layer_countdown = countdown * ceil((input_shape[2]+layer['padding'][0]+layer['padding'][1]) / layer['stride'][1])
            elif layer_type == "matmul":
                total_layer_countdown = countdown
            # import ipdb as pdb; pdb.set_trace()
            t.add_row([iShape, fShape, ilength, ijump, wlength, wjump, countdown, total_layer_countdown])
            input_shape = self.infer_activation_shape(input_shape, fShape, padding, stride, layer_type)
            total_cycles += total_layer_countdown
        self.__gen__riscv_code(self.model, _func_dict)
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
        iprec,wprec,oprec = self.prec
        for tensor_name, tensor in weight_dict.items():
            export_tensor(tensor, format="linear", prec=wprec, tensor_name=tensor_name, numerical_system=None)

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
                # import ipdb as pdb; pdb.set_trace()
                weights = layer['weight'].transpose()
                # weights = layer['weight']
            # The accelerator only works with integer values
            flatten_weights = [int(val) for val in weights.flatten()]
            # print(flatten_weights)
            weight_block = np.zeros([4096, wprec],dtype=str)
            cnt = 0
            processed = False
            # Now we need to transpose the weight tensor into MVU format.
            for idx, val in enumerate(flatten_weights):
                if cnt >= 4096 or idx==(len(flatten_weights)-1):
                    if idx==(len(flatten_weights)-1):
                        # import ipdb as pdb; pdb.set_trace()
                        weight_block[cnt] = self.int2bit(val, wprec)
                    cnt = 0
                    # import ipdb as pdb; pdb.set_trace()
                    for weight in weight_block.transpose(1,0):
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
        
