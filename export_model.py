"""Builds a small Transformer encoder in PyTorch and exports it to ONNX.

Weights are random: this project measures inference *speed*, not accuracy, so
trained weights are not needed (and downloading them is not required).
"""
import torch
import torch.nn as nn

SEQ_LEN = 128
D_MODEL = 512
N_HEAD = 8
D_FF = 2048
N_LAYERS = 2


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYERS, enable_nested_tensor=False)

    def forward(self, x):
        return self.enc(x)


def main(path="model.onnx"):
    torch.manual_seed(0)
    # The fused "fast path" produces an op ONNX cannot export; use the plain ops.
    torch.backends.mha.set_fastpath_enabled(False)
    model = Encoder().eval()
    x = torch.randn(1, SEQ_LEN, D_MODEL)
    with torch.no_grad():
        torch.onnx.export(
            model, (x,), path, input_names=["x"], output_names=["y"],
            opset_version=17, dynamo=False,
        )
        ref = model(x).numpy()
    import numpy as np
    np.save("ref_input.npy", x.numpy())
    np.save("ref_output.npy", ref)
    print(f"exported {path}: input {tuple(x.shape)}")


if __name__ == "__main__":
    main()
