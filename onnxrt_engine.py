
import onnxruntime as onnxrt
from onnxruntime import get_device
import onnxruntime.backend as backend
import time
import numpy as np

class OnnxrtEngine():
    """ONNX Runtime framework.
    
    Cross-platform inferencing engine for .onnx models. Supports CPU, GPU
    architectures.
    """
    def __init__(self, model):
        try:
            self.model     = model
            self.onnxrt    = onnxrt
            options        = self.onnxrt.SessionOptions()
            self.sess      = self.onnxrt.InferenceSession(model, options)
        except ImportError:
            print("onnxruntime package is not installed")
            return

    def run_inference(self, input_tensor):
        # import ipdb as pdb; pdb.set_trace()
        input_name = self.sess.get_inputs()[0].name
        # import ipdb as pdb; pdb.set_trace()
        inf_start = time.time()
        res       = self.sess.run(None, { input_name: input_tensor })
        inf_end   = time.time()
        elapsed_time = inf_end - inf_start
        return res, elapsed_time
