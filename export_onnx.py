from trainer import Trainer
import argparse
import os

import torch
import onnx
import onnxruntime
import onnxruntime.tools
import onnxoptimizer

import time

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="model", help="Directory to save the checkpoints in.")
parser.add_argument("--IMG_SIZE", default=64, type=int, help="Height and width of the images to train with.")
parser.add_argument("--BATCH_SIZE", default=2, type=int, help="Batch count for the training process.")
parser.add_argument("--START_B", default=0.0001, type=float, help="Beta at the first timestep.")
parser.add_argument("--END_B", default=0.02, type=float, help="Beta at the last timestep.")
args = parser.parse_args()
print(onnx.__version__, " opset=", onnx.defs.onnx_opset_version())

def test_inference_onnx(model_path):
    # Load the optimized ONNX model and test it with onnxruntime
    session = onnxruntime.InferenceSession(output_path)
    input_data = {
        session.get_inputs()[0].name: dummy_img.numpy(),
        session.get_inputs()[1].name: dummy_timestep.numpy()
    }
    start = time.time()
    output_data = session.run(None, input_data)
    end = time.time()
    time_diff = end - start
    print("inference time:")
    print(time_diff)
    print("__________________________")
    return time_diff

trainer = Trainer(args.model_dir, 0, args.START_B, args.END_B, args.IMG_SIZE, args.BATCH_SIZE)
trainer.load_checkpoint()
model = trainer.model
model.eval()

# Create dummy inputs for tracing the model.
dummy_img = torch.rand((1, 3, args.IMG_SIZE, args.IMG_SIZE)).float()
dummy_timestep = torch.randint(0, 100, (1,)).long()

print("testing pytorch model:")
start = time.time()
model(dummy_img, dummy_timestep)
end = time.time()
time_diff = end - start
print("inference time:")
print(time_diff)
print("__________________________")

output_path = "./model/model.onnx"
torch.onnx.export(model,
                (dummy_img, dummy_timestep),
                output_path,
                input_names=["img_input", "timestep_input"],
                # opset_version=11,
                output_names=["noise_prediction"]

                )


# Test unoptimized model:
print("testing onnx model:")
base_time = test_inference_onnx(output_path)
if True:
    # Optimize the ONNX model
    onnx_model = onnx.load(output_path)
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    onnx.save(optimized_model, output_path)

    print("testing optimized onnx model:")
    opt_time = test_inference_onnx(output_path)

    ratio = (opt_time / base_time) * 100
    print(f"Optimized time percentage: {ratio}")

try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print("The model is invalid: %s" % e)
else:
    print("The model is valid!")

# WebGL check:
sess = onnxruntime.InferenceSession(output_path)
# Get the list of unsupported operators used by the model
provider = onnxruntime.get_available_providers()
print(provider)
print("sodone")
# if 'WebGLExecutionProvider' in provider:
#     unsupported_ops = sess.get_providers(['WebGLExecutionProvider'])[0].get_unsupported_node_count(sess.get_inputs())
#     if unsupported_ops > 0:
#         print(f"The model uses {unsupported_ops} unsupported operators")
#     else:
#         print("The model is compatible with the WebGL backend")
# else:
#     print("The WebGLExecutionProvider is not available in this installation of ONNX Runtime")