import torch
from maia2 import model

maia2_rapid, checkpoint = model.from_pretrained(type="blitz", device="cpu")
state_dict = {
    k.replace("module.", ""): v
    for k, v in checkpoint["model_state_dict"].items()
}
maia2_rapid.load_state_dict(state_dict)
maia2_rapid.eval()

dummy_input1 = torch.randn(1, 18, 8, 8)
dummy_input2 = torch.Tensor([0]).long()
dummy_input3 = torch.Tensor([0]).long()

torch.onnx.export(
    maia2_rapid,  # Model being run
    (dummy_input1, dummy_input2, dummy_input3),  # Model inputs as a tuple
    "./maia_blitz_onnx.onnx",  # Where to save the model (can be a file or file-like object)
    export_params=True,  # Store the trained parameter weights inside the model file
    opset_version=11,  # The ONNX version to export the model to
    do_constant_folding=True,  # Whether to execute constant folding for optimization
    input_names=["boards", "elo_self", "elo_oppo"],  # The model's input names
    output_names=[
        "logits_maia",
        "logits_side_info",
        "logits_value",
    ],  # The model's output names
    dynamic_axes={
        "boards": {0: "batch_size"},
        "elo_self": {0: "batch_size"},
        "elo_oppo": {0: "batch_size"},
        "logits_maia": {0: "batch_size"},
        "logits_side_info": {0: "batch_size"},
        "logits_value": {0: "batch_size"},
    },  # Variable length axes
)
