import json
from maia2 import model, dataset, inference

maia2_model = model.from_pretrained(type="rapid", device="cpu")

prepared = inference.prepare()

move_probs, win_prob = inference.inference_each(
    maia2_model,
    prepared,
    "5B2/1n6/K1P4N/P2q4/4Br1N/3Pn2R/k2p2P1/8 w - - 0 1",
    1100,
    1100,
)
print(win_prob)
print(move_probs)
