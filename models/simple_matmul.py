import numpy as np
import torch
import torch.nn as nn

class SimpleMatMul(nn.Module):

    def __init__(self, in_ch, out_ch, wprec):
        super(SimpleMatMul, self).__init__()

        self.linear = nn.Linear(in_ch, out_ch, bias=False)

        # import ipdb as pdb; pdb.set_trace()
        max_int = (2**wprec) - 1 
        w_data = np.random.randint(max_int+1, size=(in_ch*out_ch))
        weights = np.asarray(w_data).astype(np.float32).reshape(out_ch, in_ch)
        self.linear.weight.data = torch.from_numpy(weights)

    def forward(self, x):
        out = self.linear(x)
        return out

def export_torch_to_onnx(model, batch_size, nb_channels, w, h, iprec):
    if isinstance(model, torch.nn.Module):
        model_name =  model.__class__.__name__
        # create the imput placeholder for the model
        # note: we have to specify the size of a batch of input images
        max_int = (2**iprec) - 1 
        # input_placeholder = torch.randint(0, max_int, [batch_size, nb_channels, w,h]).type(torch.int32) 
        input_placeholder = torch.randn(batch_size, nb_channels, w, h)
        # import ipdb as pdb; pdb.set_trace()
        onnx_model_fname = model_name + ".onnx"
        # export pytorch model to onnx
        torch.onnx.export(model, input_placeholder, onnx_model_fname)
        print("{0} was exported to onnx: {1}".format(model_name, onnx_model_fname))
        return onnx_model_fname
    else:
        print("Unsupported model file")
        return

if __name__ == '__main__':
    input_size = 64
    in_ch = 64
    out_ch= 64
    wprec = 2
    iprec = 2

    model = SimpleMatMul(in_ch, out_ch, wprec)
    # import ipdb as pdb; pdb.set_trace()
    # input_tensor = torch.randint(0,max_int, [1, in_ch, input_size,input_size]).type(torch.int32) 
    # # print(model(input_tensor))
    onnx_model_fname = export_torch_to_onnx(model, 1, 1, 200, input_size, iprec)
    
