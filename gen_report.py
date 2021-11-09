from math import ceil
from typing import Pattern
from texttable import Texttable
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


wprec = [1,2,3,4,5,6,7,8]
# aprec = [1,1,1,1,1,1,1,1]
aprec = [1,2,3,4,5,6,7,8]
cntd_val = np.zeros((len(wprec), len(aprec)))
oprec = 8
kernel = [1, 3, 5, 7]
# kernel = [1,3, 5]
input = [32, 64, 128, 256, 512]
# input = [64]
ichannel = [64, 128, 256, 512]
ochannel = [64, 128, 256, 512]
# ochannel = ichannel = [128]
# ichannel = [64, 128, 256, 512]
stride = [1]
mvu_var_dict = OrderedDict()
mvu_var_dict = {"ichannel" : ichannel,
                "ochannel" : ochannel ,
                "input" : input ,
                "kernel" : kernel ,
                "wprec": wprec,
                "aprec" : aprec ,
                }

VARIABLE_TO_PLOT = "input"
def get_complexity(prec, iShape, fShape, stride):
    # import ipdb as pdb; pdb.set_trace()
    iprec,wprec,oprec = prec
    oC, iC, iH, iW = iShape
    fC, fH, fW = fShape
    sW = stride[0]
    ilength = [0,0,0,0,0]
    ijump   = [0,0,0,0,0]
    wlength = [0,0,0,0,0]
    wjump   = [0,0,0,0,0]

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
    return int(countdown)

def plot3d(dicts, plot_var):
    W,A = np.meshgrid(np.asarray(wprec),np.asarray(aprec))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(10, 90)
    mycmap = plt.get_cmap('gist_earth')

    def get_pattern(new_var):
        pattern = []
        dictIndex = {k:i for i,k in enumerate(mvu_var_dict.keys())}
        for key, val in mvu_var_dict.items():
            if key == plot_var:
                pattern.append(new_var)
            elif key == "wprec" or key == "aprec":
                pass
            else:
                pattern.append(val[0])
        return pattern

    for val in mvu_var_dict[plot_var]:
        # import ipdb as pdb; pdb.set_trace()
        def get_func(dicts, template):
            Z = np.zeros((len(wprec), len(aprec)))
            pattern = template.copy()
            for i,x in enumerate(W[0]):
                for j,y in enumerate(A[0]):
                    pattern.append(x)
                    pattern.append(y)
                    Z[i,j] = dicts[repr(pattern)]
                    pattern = template.copy()
            return Z
        # import ipdb as pdb; pdb.set_trace()
        pattern = get_pattern(val)
        Z = get_func(dicts, pattern)
        # surf = ax.scatter(W, A, Z, label='Input Channel Size {0}'.format(key))
        surf = ax.plot_surface(W, A, Z, label='{0} Size {1}'.format(plot_var, val))
        surf._facecolors2d = surf._facecolor3d 
        surf._edgecolors2d = surf._edgecolor3d

        # ax.scatter(W, A, Z)
    ax.set_xlabel('wprec')
    ax.set_ylabel('aprec')
    ax.set_zlabel('countdown')
    ax.legend()
    plt.show()
    # fig.savefig('temp.png', dpi=500)


if __name__ == '__main__':
    t = Texttable(max_width=160)
    t.add_row(['wprec', 'aprec', 'kernel', 'ishape', 'input channel', 'output channel', 'countdown'])
    cnt_dict = {}
    # import ipdb as pdb; pdb.set_trace()
    for ic in ichannel:
        for oc in ochannel:
            for i in input:
                for k in kernel:
                    for w in wprec:
                        for a in aprec:
                            prec = [a,w,oprec]
                            iShape = [ceil(oc/64), ceil(ic/64), i, i]
                            fShape = [ceil(oc/64), k, k]
                            cntd = get_complexity(prec, iShape, fShape, stride)
                            cnt_dict[repr([ic,oc,i,k,w,a])] = cntd
                            # print(cntd)
                            t.add_row([w, a, (k,k), (i,i), ic, oc, cntd])
    print(t.draw())
    plot3d(cnt_dict, VARIABLE_TO_PLOT)
