import json

with open("./logits.json", "r") as f:
    probs = json.load(f)

for i, prob in enumerate(probs[0]):
    print(i, prob)
