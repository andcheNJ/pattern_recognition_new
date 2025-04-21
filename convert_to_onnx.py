# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:59:10 2025

@author: Andrew
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.model import SiameseNetwork  # <-- your existing model definition
import os

def export_siamese_model_to_onnx(
    model_path= r"C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project_1\models\siamese_final_model.pth",
    onnx_path="siamese_network.onnx",
    input_size=(1, 64, 64),  # (channels, height, width)
    opset_version=11 # Or try 13/14 if needed
):
    """
    Load the PyTorch SiameseNetwork, then export it to an ONNX file with dynamic batch size.
    """
    # 1. Load the model
    print(f"Loading model from {model_path}...")
    model = SiameseNetwork()
    # Ensure the model is on CPU for export consistency
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")

    # 2. Create dummy inputs
    #    Use batch size 1 for the dummy input, but define the 0-th axis as dynamic
    dummy_input1 = torch.randn(1, *input_size, requires_grad=False)
    dummy_input2 = torch.randn(1, *input_size, requires_grad=False)
    print(f"Dummy input shapes: {dummy_input1.shape}, {dummy_input2.shape}")

    # 3. Export to ONNX
    print(f"Exporting model to {onnx_path}...")
    try:
        torch.onnx.export(
            model,
            (dummy_input1, dummy_input2), # Tuple of inputs
            onnx_path,
            export_params=True,        # Store the trained parameter weights inside the model file
            opset_version=opset_version,   # The ONNX version to export the model to
            do_constant_folding=True,  # Whether to execute constant folding for optimization
            input_names=["input1", "input2"],   # The model's input names
            output_names=["output1", "output2"], # The model's output names
            dynamic_axes={             # Enable dynamic batch size
                "input1": {0: "batch_size"},
                "input2": {0: "batch_size"},
                "output1": {0: "batch_size"},
                "output2": {0: "batch_size"},
            },
        )
        print("-" * 60)
        print(f"Successfully exported Siamese model to {onnx_path}")
        print("-" * 60)

        # Optional: Verify the ONNX model (requires onnx and onnxruntime)
        # try:
        #     import onnx
        #     import onnxruntime as ort
        #
        #     onnx_model = onnx.load(onnx_path)
        #     onnx.checker.check_model(onnx_model)
        #     print("ONNX model check passed.")
        #
        #     # Check runtime inference (optional)
        #     # ort_session = ort.InferenceSession(onnx_path)
        #     # test_input1 = np.random.randn(2, *input_size).astype(np.float32) # Example batch size 2
        #     # test_input2 = np.random.randn(2, *input_size).astype(np.float32)
        #     # ort_inputs = {'input1': test_input1, 'input2': test_input2}
        #     # ort_outs = ort_session.run(None, ort_inputs)
        #     # print(f"ONNX Runtime output shapes: {ort_outs[0].shape}, {ort_outs[1].shape}")
        #
        # except ImportError:
        #     print("ONNX or ONNXRuntime not installed. Skipping verification.")
        # except Exception as e:
        #     print(f"ONNX verification failed: {e}")

    except Exception as e:
        print(f"Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    export_siamese_model_to_onnx()