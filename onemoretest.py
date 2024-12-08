import maia2.utils
import torch
import maia2.inference
import onnx
import maia2
import chess
import onnxruntime as ort

board = chess.Board(fen="8/4pqP1/2Kp1Pk1/2bB3R/n2Pb2p/7N/3R4/8 w - - 0 1")
board_tensor = torch.tensor(maia2.utils.board_to_tensor(board))
elo_self = torch.tensor([1100])
elo_oppo = torch.tensor([1100])


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )


# Add batch dimension to board tensor (unsqueeze at dim 0)
boards = to_numpy(
    board_tensor.unsqueeze(0)
)  # Shape: [1, channels, height, width]

# Reshape ELO inputs to rank 1
elos_self = to_numpy(torch.tensor([1]))  # Remove extra brackets
elos_oppo = to_numpy(torch.tensor([1100]))  # Remove extra brackets

ort_session = ort.InferenceSession("./maia_rapid_onnx.onnx")
ort_inputs = {
    "boards": boards,
    "elo_self": elos_self,
    "elo_oppo": elos_oppo,
}
ort_outs = ort_session.run(None, ort_inputs)
