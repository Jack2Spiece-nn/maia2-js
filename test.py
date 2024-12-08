import json
from maia2 import inference

data = inference.get_all_possible_moves_dict()

with open("all_moves_dict.json", "w") as f:
    json.dump(data, f)
