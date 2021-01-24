
class Generator():
    """docstring for Generator"""
    def __init__(self, model):
        super(Generator, self).__init__()
        # expecting to receive a OnnxModel parsed object
        self.model = model
        self.iShape = model[]
        self.fShape = model[]
        self.stride = model[]

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

    def get_mvu_param(self, prec):
        iprec,wprec,oprec = prec
        iW,iH,iC = iShape
        fW,fH,fC = fShape
        sW = stride 
        ilength[0] = iC*Fw-1
        ilength[1] = fH-1
        ilength[2] = iprec*wprec*fC-1
        ilength[3] = 0
        ijump[0]   = iprec*(iC*(iW-fW) + 1)
        ijump[1]   = -iprec*(C*(fH-1)*iW + fW*iC - 1)
        ijump[2]   = -iprec*(C*(fH-1)*iW + (fW-sW-1)*iC + 1)
        ijump[3]   = ijump2
        wlength[0] = iC*fW*fH-1
        wlength[1] = iprec*wprec-1
        wlength[2] = fC-1
        wlength[3] = 0
        wjump[0] = -wprec*(iC*fW*fH-1)
        wjump[1] = wprec
        wjump[2] = -wprec*(iC*fW*fH*fC-1)
        wjump[3] = wjump2
        countdown = (iC * fW) * (fH) * (iprec * wprec) * (fC) * ((W-fW+1)/sW)
        return [ilength, ijump, wlength, wjump, countdown]

    def print_mvU_param(self):
        pass
