import * as ort from "onnxruntime-web";

import {
  softmax,
  preprocess,
  mirrorMove,
  allPossibleMovesReversed,
} from "./utils";

class Maia {
  public Ready: Promise<boolean>;
  private model!: ort.InferenceSession;
  private options: { modelPath: string };

  constructor(options: { modelPath: string }) {
    this.options = options;
    this.Ready = new Promise(async (resolve, reject) => {
      try {
        this.model = await ort.InferenceSession.create(options.modelPath);
      } catch (e) {
        reject(e);
      }
    });
  }

  async evaluate(fen: string, eloSelf: number, eloOppo: number) {
    // Preprocess inputs
    const {
      boardInput,
      eloSelf: eloSelfCategory,
      eloOppo: eloOppoCategory,
      legalMoves,
    } = preprocess(fen, eloSelf, eloOppo);

    // Create tensors for model input
    const boardTensor = new ort.Tensor("float32", boardInput, [1, 18, 8, 8]);
    // console.log(boardTensor);
    const eloSelfTensor = new ort.Tensor(
      "int64",
      new BigInt64Array([BigInt(eloSelfCategory)]),
      [1]
    );
    const eloOppoTensor = new ort.Tensor(
      "int64",
      new BigInt64Array([BigInt(eloOppoCategory)]),
      [1]
    );

    // Load and run the model
    const feeds = {
      boards: boardTensor,
      elo_self: eloSelfTensor,
      elo_oppo: eloOppoTensor,
    };
    const { logits_maia, logits_value } = await this.model.run(feeds);

    // Process logits

    const logitsMaiaArray = logits_maia.data as Float32Array;
    const logitsMaiaLegal = logitsMaiaArray.map(
      (value, index) => value * legalMoves[index]
    );

    // Compute probabilities

    const probs = softmax(logitsMaiaLegal);
    // console.log(probs);

    // Win probability
    let winProb = Math.max(0, Math.min(1, logits_value.data[0] / 2 + 0.5));
    let blackFlag = false;
    if (fen.split(" ")[1] === "b") {
      winProb = 1 - winProb;
      blackFlag = true;
    }

    // Map move probabilities
    const legalMoveIndices = legalMoves
      .map((value, index) => (value > 0 ? index : -1))
      .filter((index) => index !== -1);

    const legalMovesMirrored: string[] = [];

    for (const moveIdx of legalMoveIndices) {
      let move = allPossibleMovesReversed[moveIdx];
      if (blackFlag) move = mirrorMove(move);

      legalMovesMirrored.push(move);
    }

    const moveProbs = {};
    legalMoveIndices.forEach((moveIdx, i) => {
      moveProbs[legalMovesMirrored[i]] = parseFloat(probs[moveIdx].toFixed(4));
    });

    // Sort moves by probability
    const sortedMoveProbs = Object.fromEntries(
      Object.entries(moveProbs).sort(([, a], [, b]) => b - a)
    );

    const top10Indices = Array.from(probs)
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10)
      .map(({ index }) => index);

    console.log("Top 10 Moves:");
    top10Indices.forEach((index) => {
      console.log(allPossibleMovesReversed[index], probs[index]);
    });

    return {
      moveProbs: sortedMoveProbs,
      winProb: parseFloat(winProb.toFixed(4)),
    };
  }
}

export default Maia;
