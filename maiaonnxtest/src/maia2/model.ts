import * as ort from "onnxruntime-web";

import {
  softmax,
  eloDict,
  mapToCategory,
  preprocess,
  allPossibleMovesReversed,
  boardToTensor,
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
    const { legalMoves } = preprocess(fen, eloSelf, eloOppo);

    const eloSelfCategory = mapToCategory(eloSelf, eloDict);
    const eloOppoCategory = mapToCategory(eloOppo, eloDict);
    const boardTensor = boardToTensor(fen);

    // Create tensors for model input
    // const boardTensor = new ort.Tensor("float32", boardInput, [1, 18, 8, 8]);
    // console.log(boardTensor);
    // const eloSelfTensor = new ort.Tensor(
    //   "int64",
    //   new BigInt64Array([BigInt(eloSelfCategory)]),
    //   [1]
    // );
    // const eloOppoTensor = new ort.Tensor(
    //   "int64",
    //   new BigInt64Array([BigInt(eloOppoCategory)]),
    //   [1]
    // );

    // Load and run the model
    const feeds: Record<string, ort.Tensor> = {
      boards: new ort.Tensor("float32", boardTensor, [1, 18, 8, 8]),
      elo_self: new ort.Tensor(
        "int64",
        BigInt64Array.from([BigInt(eloSelfCategory)])
      ),
      elo_oppo: new ort.Tensor(
        "int64",
        BigInt64Array.from([BigInt(eloOppoCategory)])
      ),
    };
    const { logits_maia } = await this.model.run(feeds);

    const probs = processOutputs(logits_maia, legalMoves);

    const logitsArray = Array.from(logits_maia.data as Float32Array);
    const logitsMaiaLegal = logitsArray.map((logit, idx) =>
      legalMoves[idx] > 0 ? logit : 0
    );

    const boardTensorArray = Array.from(boardTensor);
    console.log(boardTensorArray);
    console.log(logitsArray);
    console.log(logitsMaiaLegal);

    // Process logits

    // const logitsMaiaArray = logits_maia.data as Float32Array;
    // const logitsMaiaLegal = logitsMaiaArray.map(
    //   (value, index) => value * legalMoves[index]
    // );

    // // Compute probabilities

    // const probs = softmax(logitsMaiaLegal);
    // // console.log(probs);

    // // Win probability
    // let winProb = Math.max(0, Math.min(1, logits_value.data[0] / 2 + 0.5));
    // let blackFlag = false;
    // if (fen.split(" ")[1] === "b") {
    //   winProb = 1 - winProb;
    //   blackFlag = true;
    // }

    // // Map move probabilities
    // const legalMoveIndices = legalMoves
    //   .map((value, index) => (value > 0 ? index : -1))
    //   .filter((index) => index !== -1);

    // const legalMovesMirrored: string[] = [];

    // for (const moveIdx of legalMoveIndices) {
    //   let move = allPossibleMovesReversed[moveIdx];
    //   if (blackFlag) move = mirrorMove(move);

    //   legalMovesMirrored.push(move);
    // }

    // const moveProbs = {};
    // legalMoveIndices.forEach((moveIdx, i) => {
    //   moveProbs[legalMovesMirrored[i]] = parseFloat(probs[moveIdx].toFixed(4));
    // });

    // // Sort moves by probability
    // const sortedMoveProbs = Object.fromEntries(
    //   Object.entries(moveProbs).sort(([, a], [, b]) => b - a)
    // );

    // const top10Indices = Array.from(logitsMaiaLegal)
    //   .map((value, index) => ({ value, index }))
    //   .sort((a, b) => b.value - a.value)
    //   .slice(0, 10)
    //   .map(({ index }) => index);

    // console.log("Top 10 Moves:");
    // top10Indices.forEach((index) => {
    //   console.log(
    //     allPossibleMovesReversed[index],
    //     probs[index],
    //     logitsMaiaLegal[index]
    //   );
    // });

    return {
      moveProbs: probs,
      // winProb: parseFloat(winProb.toFixed(4)),
    };
  }
}

function processOutputs(output: ort.Tensor, legalMoves: number[]) {
  const logits = output.data as Float32Array;

  // Get indices of legal moves
  const legalMoveIndices = legalMoves
    .map((value, index) => (value > 0 ? index : -1))
    .filter((index) => index !== -1);

  // Extract logits for legal moves
  const legalLogits = legalMoveIndices.map((idx) => logits[idx]);

  // Compute softmax over the legal logits
  const maxLogit = Math.max(...legalLogits);
  const expLogits = legalLogits.map((logit) => Math.exp(logit - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  const probs = expLogits.map((expLogit) => expLogit / sumExp);

  // Map the probabilities back to their move indices
  const moveProbs = {};
  for (let i = 0; i < legalMoveIndices.length; i++) {
    moveProbs[allPossibleMovesReversed[legalMoveIndices[i]]] = probs[i];
  }

  return moveProbs;
}

export default Maia;
