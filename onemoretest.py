import onnx
from pprint import pprint
import torch
import chess
import numpy as np
import onnxruntime as ort
from maia2 import inference


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )


def mirror_move(move_uci):
    # Check if the move is a promotion (length of UCI string will be more than 4)
    is_promotion = len(move_uci) > 4

    # Extract the start and end squares, and the promotion piece if applicable
    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    promotion_piece = move_uci[4:] if is_promotion else ""

    # Mirror the start and end squares
    mirrored_start = mirror_square(start_square)
    mirrored_end = mirror_square(end_square)

    # Return the mirrored move, including the promotion piece if applicable
    return mirrored_start + mirrored_end + promotion_piece


def mirror_square(square):

    file = square[0]
    rank = str(9 - int(square[1]))

    return file + rank


all_moves = inference.get_all_possible_moves()
all_moves_dict = {move: i for i, move in enumerate(all_moves)}
all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
elo_dict = inference.create_elo_dict()

board = chess.Board(fen=input("Enter FEN: "))
board_input, elo_self, elo_oppo, legal_moves = inference.preprocessing(
    board.fen(), 1100, 1100, elo_dict, all_moves_dict
)

boards2 = to_numpy(torch.tensor(inference.board_to_tensor(board)).unsqueeze(0))

boards = board_input.unsqueeze(0)
elos_self = to_numpy(torch.tensor([elo_self]))
elos_oppo = to_numpy(torch.tensor([elo_oppo]))

ort_session = ort.InferenceSession("./maia_rapid_onnx.onnx")
ort_inputs = {
    "boards": boards2,
    "elo_self": elos_self,
    "elo_oppo": elos_oppo,
}
ort_outs = ort_session.run(None, ort_inputs)
logits_maia_legal = ort_outs[0] * legal_moves.numpy()
np.save("board_onnx_py.npy", boards2)
np.save("logits_python.npy", ort_outs[0])
np.save("logits_python_legal.npy", logits_maia_legal)
probs = torch.tensor(logits_maia_legal).softmax(dim=-1).cpu().tolist()
preds = np.argmax(logits_maia_legal, axis=-1)


black_flag = False
if board.fen().split(" ")[1] == "b":
    # logits_value = 1 - logits_value
    black_flag = True


move_probs = {}
legal_move_indices = legal_moves.nonzero().flatten().cpu().numpy().tolist()
legal_moves_mirrored = []
for move_idx in legal_move_indices:
    move = all_moves_dict_reversed[move_idx]
    if black_flag:
        move = mirror_move(move)
    legal_moves_mirrored.append(move)

for j in range(len(legal_move_indices)):
    move_probs[legal_moves_mirrored[j]] = round(
        probs[0][legal_move_indices[j]], 4
    )

move_probs = dict(
    sorted(move_probs.items(), key=lambda item: item[1], reverse=True)
)

print(move_probs)
