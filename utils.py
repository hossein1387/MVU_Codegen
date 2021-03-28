import numpy as np

def export_tensor(dict):
    for key, vals in dict.items():
        file_name = "{}.hex".format(key)
        lines = ""
        with open(file_name, "w") as f:
            for val in vals:
                # import ipdb as pdb; pdb.set_trace()
                lines += "{}\n".format(val)
            lines = lines[:-1]
            f.write(lines)
        f.close()
        print("Exporting {} to {}".format(key, file_name))

def gen_test_vecs(model_path, precision, input_shape):
    import onnxrt_engine as onnxrt
    # import ipdb as pdb; pdb.set_trace()
    onnxrt_engine = onnxrt.OnnxrtEngine(model_path)
    input_tensor = gen_input_test_vectors(precision, input_shape)
    output, time = onnxrt_engine.run_inference(input_tensor)
    print("Inference finised in {:4.4f} seconds".format(time))
    # import ipdb as pdb; pdb.set_trace()
    export_tensor({"output":output[0].astype(np.int32).flatten(), "input":input_tensor.astype(np.int32).flatten()})

def gen_input_test_vectors(precision, input_shape):
    # import ipdb as pdb; pdb.set_trace()
    aprec = precision[0]
    max_int = (2**aprec) - 1
    input_data = np.random.randint(max_int+1, size=(input_shape))
    input_tensor = np.asarray(input_data).astype(np.float32).reshape(input_shape)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor