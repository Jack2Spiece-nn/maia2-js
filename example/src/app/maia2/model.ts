import * as ort from "onnxruntime-web";

import { mirrorMove, preprocess, allPossibleMovesReversed } from "./utils";

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
    const { boardInput, legalMoves, eloSelfCategory, eloOppoCategory } =
      preprocess(fen, eloSelf, eloOppo);

    // Load and run the model
    const feeds: Record<string, ort.Tensor> = {
      boards: new ort.Tensor("float32", boardInput, [1, 18, 8, 8]),
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

    const probs = processOutputs(fen, logits_maia, legalMoves);

    return {
      moveProbs: probs,
    };
  }
}

function processOutputs(
  fen: string,
  output: ort.Tensor,
  legalMoves: Float32Array
) {
  const logits = output.data as Float32Array;

  let black_flag = false;
  if (fen.split(" ")[1] === "b") {
    black_flag = true;
  }

  // Get indices of legal moves
  const legalMoveIndices = legalMoves
    .map((value, index) => (value > 0 ? index : -1))
    .filter((index) => index !== -1);

  const legalMovesMirrored = [];
  for (const moveIndex of legalMoveIndices) {
    let move = allPossibleMovesReversed[moveIndex];
    if (black_flag) {
      move = mirrorMove(move);
    }

    legalMovesMirrored.push(move);
  }

  // Extract logits for legal moves
  const legalLogits = legalMoveIndices.map((idx) => logits[idx]);

  // Compute softmax over the legal logits
  const maxLogit = Math.max(...legalLogits);
  const expLogits = legalLogits.map((logit) => Math.exp(logit - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  const probs = expLogits.map((expLogit) => expLogit / sumExp);

  // Map the probabilities back to their move indices
  const moveProbs: Record<string, number> = {};
  for (let i = 0; i < legalMoveIndices.length; i++) {
    moveProbs[legalMovesMirrored[i]] = probs[i];
  }
  // for (let i = 0; i < legalMoveIndices.length; i++) {
  //   moveProbs[allPossibleMovesReversed[legalMoveIndices[i]]] = probs[i];
  // }

  return moveProbs;
}

export default Maia;
