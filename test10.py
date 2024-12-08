import maia2
import chess
import numpy as np
from maia2 import model, inference


maia2_model = model.from_pretrained(type="rapid", device="cpu")

prepared = inference.prepare()
fen = "3r4/pb3R2/p1n5/1p1p4/2PN4/5R1P/3pN3/1K2k3 w - - 0 1"

move_prob, win_prob, logits_maia = inference.inference_each(
    maia2_model,
    prepared,
    fen,
    1100,
    1110,
)

np.save("maia2_raw_logits.npy", logits_maia.detach().numpy())


np.save(
    "board_maia.npy",
    inference.board_to_tensor(chess.Board(fen)),
)


print(move_prob)
