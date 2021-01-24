from texttable import Texttable

class Generator():
    """docstring for Generator"""
    def __init__(self, model, prec, input_shape):
        super(Generator, self).__init__()
        # expecting to receive a OnnxModel parsed object
        self.model = model
        self.prec = prec
        self.input_shape = input_shape

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

    def get_mvu_param(self, prec, iShape, fShape, stride):
        iprec,wprec,oprec = prec
        iC, iH, iW = iShape
        fC, fH, fW = fShape
        sW = stride 
        ilength = [0,0,0,0]
        ijump = [0,0,0,0]
        wlength = [0,0,0,0]
        wjump = [0,0,0,0]
        ilength[0] = iC*fW-1
        ilength[1] = fH-1
        ilength[2] = iprec*wprec*fC-1
        ilength[3] = 0
        ijump[0]   = iprec*(iC*(iW-fW) + 1)
        ijump[1]   = -iprec*(iC*(fH-1)*iW + fW*iC - 1)
        ijump[2]   = -iprec*(iC*(fH-1)*iW + (fW-sW-1)*iC + 1)
        ijump[3]   = ijump[2]
        wlength[0] = iC*fW*fH-1
        wlength[1] = iprec*wprec-1
        wlength[2] = fC-1
        wlength[3] = 0
        wjump[0] = -wprec*(iC*fW*fH-1)
        wjump[1] = wprec
        wjump[2] = -wprec*(iC*fW*fH*fC-1)
        wjump[3] = wjump[2]
        countdown = (iC * fW) * (fH) * (iprec * wprec) * (fC) * ((iW-fW+1)/sW)
        return [ilength, ijump, wlength, wjump, countdown]

    def infer_activation_shape(self, input, kernel, padding, stride):
        iC, iH, iW = input
        fC, fH, fW = kernel
        oH=int((iH-fH+2*padding)/stride)+1
        oW=int((iW-fW+2*padding)/stride)+1
        oC=fC
        return [oC, oH, oW]

    def generate_mvu_configs(self):
        t = Texttable(max_width=160)
        t.add_row(['iShape', 'fShape', 'ilength', 'ijump', 'wlength', 'wjump', 'countdown'])
        input_shape = self.input_shape
        total_cycles = 0
        for layer in self.model.parsed_model:
            # import ipdb as pdb; pdb.set_trace()
            iShape = input_shape
            fShape = [layer['out_channels'], layer['kernel_size'][0], layer['kernel_size'][1]]
            stride = layer['stride'][0]
            padding = layer['padding'][0]
            prec = self.prec
            # print("{} * {}".format(iShape, fShape))
            ilength, ijump, wlength, wjump, countdown = self.get_mvu_param(prec, iShape, fShape, stride)
            t.add_row([iShape, fShape, ilength, ijump, wlength, wjump, countdown])
            input_shape = self.infer_activation_shape(input_shape, fShape, padding, stride)
            total_cycles += countdown
        print("\nGenerated MVU configuration:")
        print(t.draw())
        print("Total countdown: {}".format(int(total_cycles)))


    def print_mvU_param(self):
        pass
