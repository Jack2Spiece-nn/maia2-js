import maia2
from maia2 import model, inference


maia2_model = model.from_pretrained(type="rapid", device="cpu")

prepared = inference.prepare()

move_prob, win_prob = inference.inference_each(
    maia2_model,
    prepared,
    "1B6/3k3K/P2P2p1/6P1/2B2p2/pP1N4/1P4b1/b3n3 b - - 0 1",
    1100,
    1110,
)

print(move_prob)
