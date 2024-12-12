"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const ort = __importStar(require("onnxruntime-web"));
const utils_1 = require("./utils");
class Maia {
    constructor(options) {
        this.Ready = new Promise((resolve, reject) => __awaiter(this, void 0, void 0, function* () {
            try {
                const buffer = yield this.getCachedModel(options.modelPath);
                this.model = yield ort.InferenceSession.create(buffer);
                resolve(true);
            }
            catch (e) {
                reject(e);
            }
        }));
    }
    getCachedModel(url) {
        return __awaiter(this, void 0, void 0, function* () {
            const cache = yield caches.open("maia2-model");
            const response = yield cache.match(url);
            if (response) {
                return response.arrayBuffer();
            }
            else {
                const response = yield fetch(url);
                if (response.ok) {
                    yield cache.put(url, response.clone());
                    return response.arrayBuffer();
                }
                else {
                    throw new Error("Failed to fetch model");
                }
            }
        });
    }
    /**
     * Evaluates a given chess position using the Maia model.
     *
     * @param board - The FEN string representing the chess position.
     * @param eloSelf - The ELO rating of the player making the move.
     * @param eloOppo - The ELO rating of the opponent.
     * @returns A promise that resolves to an object containing the policy and value predictions.
     */
    evaluate(board, eloSelf, eloOppo) {
        return __awaiter(this, void 0, void 0, function* () {
            const { boardInput, legalMoves, eloSelfCategory, eloOppoCategory } = (0, utils_1.preprocess)(board, eloSelf, eloOppo);
            // Load and run the model
            const feeds = {
                boards: new ort.Tensor("float32", boardInput, [1, 18, 8, 8]),
                elo_self: new ort.Tensor("int64", BigInt64Array.from([BigInt(eloSelfCategory)])),
                elo_oppo: new ort.Tensor("int64", BigInt64Array.from([BigInt(eloOppoCategory)])),
            };
            const { logits_maia, logits_value } = yield this.model.run(feeds);
            const { policy, value } = processOutputs(board, logits_maia, logits_value, legalMoves);
            return {
                policy,
                value,
            };
        });
    }
    /**
     * Evaluates a batch of chess positions using the Maia model.
     *
     * @param boards - An array of FEN strings representing the chess positions.
     * @param eloSelfs - An array of ELO ratings for the player making the move.
     * @param eloOppos - An array of ELO ratings for the opponent.
     * @returns A promise that resolves to an array of objects containing the policy and value predictions.
     */
    batchEvaluate(boards, eloSelfs, eloOppos) {
        return __awaiter(this, void 0, void 0, function* () {
            const batchSize = boards.length;
            const boardInputs = [];
            const eloSelfCategories = [];
            const eloOppoCategories = [];
            const legalMoves = [];
            for (let i = 0; i < boards.length; i++) {
                const { boardInput, legalMoves: legalMoves_, eloSelfCategory, eloOppoCategory, } = (0, utils_1.preprocess)(boards[i], eloSelfs[i], eloOppos[i]);
                boardInputs.push(boardInput);
                eloSelfCategories.push(eloSelfCategory);
                eloOppoCategories.push(eloOppoCategory);
                legalMoves.push(legalMoves_);
            }
            const combinedBoardInputs = new Float32Array(batchSize * 18 * 8 * 8);
            for (let i = 0; i < batchSize; i++) {
                combinedBoardInputs.set(boardInputs[i], i * 18 * 8 * 8);
            }
            const feeds = {
                boards: new ort.Tensor("float32", combinedBoardInputs, [
                    batchSize,
                    18,
                    8,
                    8,
                ]),
                elo_self: new ort.Tensor("int64", BigInt64Array.from(eloSelfCategories.map(BigInt)), [batchSize]),
                elo_oppo: new ort.Tensor("int64", BigInt64Array.from(eloOppoCategories.map(BigInt)), [batchSize]),
            };
            const start = performance.now();
            const { logits_maia, logits_value } = yield this.model.run(feeds);
            const end = performance.now();
            const results = [];
            for (let i = 0; i < batchSize; i++) {
                const logitsPerItem = logits_maia.size / batchSize;
                const startIdx = i * logitsPerItem;
                const endIdx = startIdx + logitsPerItem;
                const policyLogitsArray = logits_maia.data.slice(startIdx, endIdx);
                const policyTensor = new ort.Tensor("float32", policyLogitsArray, [
                    logitsPerItem,
                ]);
                const valueLogit = logits_value.data[i];
                const valueTensor = new ort.Tensor("float32", [valueLogit], [1]);
                const { policy, value: winProb } = processOutputs(boards[i], policyTensor, valueTensor, legalMoves[i]);
                results.push({ policy, value: winProb });
            }
            return {
                result: results,
                time: end - start,
            };
        });
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
function processOutputs(fen, logits_maia, logits_value, legalMoves) {
    const logits = logits_maia.data;
    const value = logits_value.data;
    let winProb = Math.min(Math.max(value[0] / 2 + 0.5, 0), 1);
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
        let move = utils_1.allPossibleMovesReversed[moveIndex];
        if (black_flag) {
            move = (0, utils_1.mirrorMove)(move);
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
    const moveProbs = {};
    for (let i = 0; i < legalMoveIndices.length; i++) {
        moveProbs[legalMovesMirrored[i]] = probs[i];
    }
    return { policy: moveProbs, value: winProb };
}
exports.default = Maia;
