import json
from maia2 import model, dataset, inference

maia2_model = model.from_pretrained(type="rapid", device="cpu")

prepared = inference.prepare()

move_probs, win_prob = inference.inference_each(
    maia2_model,
    prepared,
    "8/4pqP1/2Kp1Pk1/2bB3R/n2Pb2p/7N/3R4/8 w - - 0 1",
    1100,
    1100,
)
print(win_prob)
print(move_probs)
