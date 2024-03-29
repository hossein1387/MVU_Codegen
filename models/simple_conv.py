import numpy as np
import torch
import torch.nn as nn

class SimpleConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, groups, dilation, wprec):
        super(SimpleConv, self).__init__()

        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)

        # import ipdb as pdb; pdb.set_trace()
        max_int = (2**wprec) - 1 
        w_data = np.random.randint(max_int+1, size=(in_ch*out_ch*kernel_size*kernel_size))
        weights = np.asarray(w_data).astype(np.float32).reshape(out_ch, in_ch,kernel_size,kernel_size)
        self.conv1.weight.data = torch.from_numpy(weights)

    def forward(self, x):
        out = self.conv1(x)
        return out

def export_torch_to_onnx(model, batch_size, nb_channels, w, h):
    if isinstance(model, torch.nn.Module):
        model_name =  model.__class__.__name__
        # create the imput placeholder for the model
        # note: we have to specify the size of a batch of input images
        input_placeholder = torch.randn(batch_size, nb_channels, w, h)
        onnx_model_fname = model_name + ".onnx"
        # export pytorch model to onnx
        torch.onnx.export(model, input_placeholder, onnx_model_fname)
        print("{0} was exported to onnx: {1}".format(model_name, onnx_model_fname))
        return onnx_model_fname
    else:
        print("Unsupported model file")
        return

if __name__ == '__main__':
    input_size = 32
    in_ch = 64
    out_ch= 64
    kernel_size = 3
    stride = 1
    padding = 1
    groups = 1
    dilation = 1
    wprec = 2
    model = SimpleConv(in_ch, out_ch, kernel_size, stride, padding, groups ,dilation, wprec)
    # import ipdb as pdb; pdb.set_trace()
    input_tensor = torch.randint(0,255, [1, in_ch, input_size,input_size]).type(torch.float32) 
    # print(model(input_tensor))
    export_torch_to_onnx(model, 1, in_ch, input_size,input_size)
