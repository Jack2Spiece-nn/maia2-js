import * as ort from "onnxruntime-web";

import { mirrorMove, preprocess, allPossibleMovesReversed } from "./utils";

class Maia {
  public Ready: Promise<boolean>;
  public model!: ort.InferenceSession;
  private options: { modelPath: string };

  constructor(options: { modelPath: string }) {
    this.options = options;
    this.Ready = new Promise(async (resolve, reject) => {
      try {
        const buffer = await this.getCachedModel(options.modelPath);
        this.model = await ort.InferenceSession.create(buffer);
        resolve(true);
      } catch (e) {
        reject(e);
      }
    });
  }

  private async getCachedModel(url: string): Promise<ArrayBuffer> {
    const cache = await caches.open("maia2-model");
    const response = await cache.match(url);
    if (response) {
      return response.arrayBuffer();
    } else {
      const response = await fetch(url);
      if (response.ok) {
        await cache.put(url, response.clone());
        return response.arrayBuffer();
      } else {
        throw new Error("Failed to fetch model");
      }
    }
  }

  /**
   * Evaluates a given chess position using the Maia model.
   *
   * @param fen - The FEN string representing the chess position.
   * @param eloSelf - The ELO rating of the player making the move.
   * @param eloOppo - The ELO rating of the opponent.
   * @returns A promise that resolves to an object containing the policy and value predictions.
   */
  async evaluate(fen: string, eloSelf: number, eloOppo: number) {
    const { boardInput, legalMoves, eloSelfCategory, eloOppoCategory } =
      preprocess(fen, eloSelf, eloOppo);

    // Load and run the model
    const feeds: Record<string, ort.Tensor> = {
      boards: new ort.Tensor("float32", boardInput, [1, 18, 8, 8]),
      elo_self: new ort.Tensor(
        "int64",
        BigInt64Array.from([BigInt(eloSelfCategory)]),
      ),
      elo_oppo: new ort.Tensor(
        "int64",
        BigInt64Array.from([BigInt(eloOppoCategory)]),
      ),
    };
    const { logits_maia, logits_value } = await this.model.run(feeds);

    const { policy, value } = processOutputs(
      fen,
      logits_maia,
      logits_value,
      legalMoves,
    );

    return {
      policy,
      value,
    };
  }
}

/**
 * Processes the outputs of the ONNX model to compute the policy and value.
 *
 * @param {string} fen - The FEN string representing the current board state.
 * @param {ort.Tensor} logits_maia - The logits tensor for the policy output from the model.
 * @param {ort.Tensor} logits_value - The logits tensor for the value output from the model.
 * @param {Float32Array} legalMoves - An array indicating the legal moves.
 * @returns {{ policy: Record<string, number>, value: number }} An object containing the policy (move probabilities) and the value (win probability).
 */
function processOutputs(
  fen: string,
  logits_maia: ort.Tensor,
  logits_value: ort.Tensor,
  legalMoves: Float32Array,
) {
  const logits = logits_maia.data as Float32Array;
  const value = logits_value.data as Float32Array;

  let winProb = Math.min(Math.max((value[0] as number) / 2 + 0.5, 0), 1);

  let black_flag = false;
  if (fen.split(" ")[1] === "b") {
    black_flag = true;
    winProb = 1 - winProb;
  }

  winProb = Math.round(winProb * 10000) / 10000;

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

  return { policy: moveProbs, value: winProb };
}

export default Maia;
