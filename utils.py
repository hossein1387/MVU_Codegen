import numpy as np
import sys


def int2bit(val, precision):
    val_str = '{0:032b}'.format(val)
    return list(val_str[::-1][0:precision][::-1])

def export_tensor_to_file(tensor, tensor_name, numerical_system, prec):
    file_name = "{}.hex".format(tensor_name)
    lines = ""
    with open(file_name, "w") as f:
        for val in tensor:
            # import ipdb as pdb; pdb.set_trace()
            if numerical_system == "bin":
                val_str = '{0:064b}'.format(val)
                val = list(val_str[::-1][0:prec][::-1])
                val = "".join(val)
            elif numerical_system == "hex":
                hex_val_str = f'{int(val, 2):X}'
                val = hex_val_str.zfill(16)
            elif numerical_system == "int":
                val = int(val)
            else:
                val = val
            lines += "{}\n".format(val)
        lines = lines[:-1]
        f.write(lines)
    f.close()
    print("Exporting {} to {}".format(tensor_name, file_name))

def export_tensor(tensor, format="linear", prec=2, tensor_name=None, numerical_system="bin"):
    # expecting input shapes is:
    # [input_channels, output_channels, width, height]
    if format == "linear":
        export_tensor_to_file(tensor, tensor_name, numerical_system, prec)
    elif format == "msb_transposed":
        # import ipdb as pdb; pdb.set_trace()
        transposed_tensor = []
        # The accelerator only works with integer values
        flatten_tensor = [int(val) for val in tensor.flatten()]
        # print(flatten_weights)
        tensor_block = np.zeros([64, prec],dtype=str)
        cnt = 0
        processed = False
        # Now we need to transpose the weight tensor into MVU format.
        for idx, val in enumerate(flatten_tensor):
            if cnt >= 64 or idx==(len(flatten_tensor)-1):
                if idx==(len(flatten_tensor)-1):
                    # import ipdb as pdb; pdb.set_trace()
                    tensor_block[cnt] = int2bit(val, prec)
                cnt = 0
                for weight in tensor_block.transpose(1,0):
                    # import ipdb as pdb; pdb.set_trace()
                    val_str = "".join(weight)
                    val_str = val_str.zfill(64)[::-1]
                    if numerical_system == "bin":
                        transposed_tensor.append(val_str)
                    elif numerical_system == "hex":
                        hex_val_str = f'{int(val_str, 2):X}'
                        transposed_tensor.append(hex_val_str.zfill(16))
                    elif numerical_system == "int":
                        transposed_tensor.append(int(val_str, 2))
                    else:
                        transposed_tensor.append(val_str)
                    processed = True
            tensor_block[cnt] = int2bit(val, prec)
            cnt += 1
        # import ipdb as pdb; pdb.set_trace()
        if not processed:
            # import ipdb as pdb; pdb.set_trace()
            for weight in tensor_block.transpose(1,0):
                val_str = "".join(weight)
                #val_str = val_str[::-1].zfill(4096)
                transposed_tensor.append(val_str)
        export_tensor_to_file(transposed_tensor, tensor_name, numerical_system=None, prec=None)
    else:
        print("Unknown format {}".format(format))
        sys.exit()



def gen_test_vecs(model_path, precision, input_shape):
    import onnxrt_engine as onnxrt
    # import ipdb as pdb; pdb.set_trace()
    iprec,wprec,oprec = precision
    onnxrt_engine = onnxrt.OnnxrtEngine(model_path)
    input_tensor = gen_input_test_vectors(precision, input_shape, diag=True)
    output, time = onnxrt_engine.run_inference(input_tensor)
    print("Inference finised in {:4.4f} seconds".format(time))
    # import ipdb as pdb; pdb.set_trace()
    export_tensor(output[0].astype(np.int32).flatten(), format="msb_transposed", prec=oprec, tensor_name="output", numerical_system="hex")
    export_tensor(input_tensor.astype(np.int32).flatten(), format="msb_transposed", prec=iprec, tensor_name="input", numerical_system="bin")

def gen_input_test_vectors(precision, input_shape, diag):
    np.random.seed(0)
    # import ipdb as pdb; pdb.set_trace()
    iprec,wprec,oprec = precision
    max_int = (2**iprec) - 1
    if diag:
        input_data = np.diag(np.ones(input_shape[1]), 0) 
    else:
        input_data = np.random.randint(max_int+1, size=(input_shape))
    input_tensor = np.asarray(input_data).astype(np.float32).reshape(input_shape)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor
