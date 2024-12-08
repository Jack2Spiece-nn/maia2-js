"use client";
import { Chess } from "chess.js";
import { createAllMovesDictReversed, createEloDict } from "./all_moves";
import allPossibleMoves from "./all_moves_dict.json";

// Mirror moves for black pieces
function mirrorMove(move) {
  const files = "abcdefgh";
  const ranks = "12345678";
  const [fromFile, fromRank, toFile, toRank] = move;
  const mirroredFrom =
    files[7 - files.indexOf(fromFile)] + ranks[8 - parseInt(fromRank)];
  const mirroredTo =
    files[7 - files.indexOf(toFile)] + ranks[8 - parseInt(toRank)];
  return mirroredFrom + mirroredTo;
}

function mirrorBoard(fen: string): string {
  const parts = fen.split(" ");
  const board = parts[0];
  const turn = parts[1] === "w" ? "b" : "w"; // Swap turn
  const castlingRights = parts[2];
  const enPassant = parts[3];

  // Mirror the board: Flip rows and swap piece colors
  const mirroredRows = board
    .split("/")
    .reverse()
    .map((row) => {
      return row
        .split("")
        .map((char) => {
          if (char >= "a" && char <= "z") {
            return char.toUpperCase(); // Swap black to white
          } else if (char >= "A" && char <= "Z") {
            return char.toLowerCase(); // Swap white to black
          } else {
            return char; // Keep numbers as-is
          }
        })
        .join("");
    });
  const mirroredBoard = mirroredRows.join("/");

  // Mirror castling rights
  const mirroredCastlingRights = castlingRights
    .replace("K", "k")
    .replace("Q", "q")
    .replace("k", "K")
    .replace("q", "Q");

  // Mirror en passant square
  const mirroredEnPassant = enPassant === "-" ? "-" : mirrorSquare(enPassant);

  // Return the new FEN
  return `${mirroredBoard} ${turn} ${mirroredCastlingRights} ${mirroredEnPassant} ${parts[4]} ${parts[5]}`;
}

function mirrorSquare(square: string): string {
  const file = square.charAt(0);
  const rank = square.charAt(1);
  const mirroredRank = (9 - parseInt(rank)).toString(); // Mirror rank (1 -> 8, 2 -> 7, etc.)
  return file + mirroredRank;
}

// Mapping Elo to category
function mapToCategory(elo: number, eloDict: Record<string, number>): number {
  const interval = 100;
  const start = 1100;
  const end = 2000;

  if (elo < start) {
    return eloDict[`<${start}`];
  } else if (elo >= end) {
    return eloDict[`>=${end}`];
  } else {
    for (let lowerBound = start; lowerBound < end; lowerBound += interval) {
      const upperBound = lowerBound + interval;
      if (elo >= lowerBound && elo < upperBound) {
        return eloDict[`${lowerBound}-${upperBound - 1}`];
      }
    }
  }
  throw new Error("Elo value is out of range.");
}

// Converting a chess board to tensor-like array
function boardToTensor(board: Chess): Float32Array {
  const pieceTypes = [
    "p", // Pawn
    "n", // Knight
    "b", // Bishop
    "r", // Rook
    "q", // Queen
    "k", // King
  ];
  const numPieceChannels = 12; // 6 piece types * 2 colors
  const additionalChannels = 6; // Turn, castling rights, en passant
  const tensor = new Float32Array(
    (numPieceChannels + additionalChannels) * 8 * 8
  );

  const pieceIndices: Record<string, number> = pieceTypes.reduce(
    (acc, piece, idx) => {
      acc[piece] = idx;
      return acc;
    },
    {}
  );

  // Fill tensor for each piece type
  for (const pieceType of pieceTypes) {
    for (const color of [true, false]) {
      const pieceColor = color ? "w" : "b";
      const index = pieceIndices[pieceType] + (color ? 0 : 6);
      board.board().forEach((row, rowIndex) => {
        row.forEach((square, colIndex) => {
          if (
            square &&
            square.type === pieceType &&
            square.color === pieceColor
          ) {
            const flatIndex = (index * 8 + rowIndex) * 8 + colIndex;
            tensor[flatIndex] = 1.0;
          }
        });
      });
    }
  }

  // Player's turn channel
  const turnChannel = numPieceChannels * 8 * 8;
  if (board.turn() === "w") {
    for (let i = 0; i < 8 * 8; i++) {
      tensor[turnChannel + i] = 1.0;
    }
  }

  // Castling rights channels
  const castlingRights = [
    board.getCastlingRights("w").k,
    board.getCastlingRights("w").q,
    board.getCastlingRights("b").k,
    board.getCastlingRights("b").q,
  ];
  for (let i = 0; i < 4; i++) {
    if (castlingRights[i]) {
      const offset = (numPieceChannels + 1 + i) * 8 * 8;
      for (let j = 0; j < 8 * 8; j++) {
        tensor[offset + j] = 1.0;
      }
    }
  }

  // En passant target channel
  const epChannel = numPieceChannels + 5;
  const fenParts = board.fen().split(" ");
  const enPassantSquare = fenParts[3]; // The en passant square is the 4th part of the FEN string

  if (enPassantSquare !== "-") {
    const col = enPassantSquare.charCodeAt(0) - "a".charCodeAt(0); // Column as 0-based index
    const row = 8 - parseInt(enPassantSquare[1], 10); // Row as 0-based index
    const flatIndex = (epChannel * 8 + row) * 8 + col;
    tensor[flatIndex] = 1.0;
  }

  return tensor;
}

function preprocess(
  fen: string,
  eloSelf: number,
  eloOppo: number,
  eloDict: Record<string, number>,
  allMovesDict: Record<string, number>
): {
  boardInput: Float32Array;
  eloSelf: number;
  eloOppo: number;
  legalMoves: Float32Array;
} {
  // Handle mirroring if it's black's turn
  let board = new Chess(fen);
  // if (fen.split(" ")[1] === "b") {
  //   board = new Chess(mirrorBoard(board.fen()));
  // } else if (fen.split(" ")[1] !== "w") {
  //   throw new Error(`Invalid FEN: ${fen}`);
  // }

  // Convert board to tensor
  const boardInput = boardToTensor(board);

  // Map Elo to categories
  const eloSelfCategory = mapToCategory(eloSelf, eloDict);
  const eloOppoCategory = mapToCategory(eloOppo, eloDict);

  // Generate legal moves tensor
  console.log(board.fen());
  const legalMoves = new Float32Array(Object.keys(allMovesDict).length);
  for (const move of board.moves({ verbose: true })) {
    console.log(move.lan);
    const moveIndex = allMovesDict[move.lan];

    if (moveIndex !== undefined) {
      legalMoves[moveIndex] = 1.0;
    }
  }

  return {
    boardInput,
    eloSelf: eloSelfCategory,
    eloOppo: eloOppoCategory,
    legalMoves,
  };
}

async function inferenceEach(modelPath, prepared, fen, eloSelf, eloOppo) {
  const [allMovesDict, eloDict, allMovesDictReversed] = prepared;

  // Preprocess inputs
  const {
    boardInput,
    eloSelf: eloSelfCategory,
    eloOppo: eloOppoCategory,
    legalMoves,
  } = preprocess(fen, eloSelf, eloOppo, eloDict, allMovesDict);

  // Create tensors for model input
  const boardTensor = new ort.Tensor("float32", boardInput, [1, 18, 8, 8]);
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
  const legalMovesTensor = new ort.Tensor("float32", legalMoves, [
    1,
    legalMoves.length,
  ]);

  // Load and run the model
  const session = await ort.InferenceSession.create(modelPath);
  const feeds = {
    boards: boardTensor,
    elo_self: eloSelfTensor,
    elo_oppo: eloOppoTensor,
  };
  const { logits_maia, logits_value } = await session.run(feeds);

  // Process logits
  const logitsMaiaArray = logits_maia.data;

  const legalMovesArray = legalMoves;
  const logitsMaiaLegal = logitsMaiaArray.map(
    (value, index) => value * legalMovesArray[index]
  );

  // Compute probabilities
  const probs = softmax(logitsMaiaLegal);

  // Win probability
  let winProb = Math.max(0, Math.min(1, logits_value.data[0] / 2 + 0.5));
  let blackFlag = false;
  if (fen.split(" ")[1] === "b") {
    winProb = 1 - winProb;
    blackFlag = true;
  }

  // Map move probabilities
  const legalMoveIndices = legalMovesArray
    .map((value, index) => (value > 0 ? index : -1))
    .filter((index) => index !== -1);

  const legalMovesMirrored: string[] = [];

  for (const moveIdx of legalMoveIndices) {
    let move = allMovesDictReversed[moveIdx];

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

  return {
    moveProbs: sortedMoveProbs,
    winProb: parseFloat(winProb.toFixed(4)),
  };
}

// Softmax function
function softmax(values) {
  const maxVal = Math.max(...values);
  const expVals = values.map((val) => Math.exp(val - maxVal));
  const sumExpVals = expVals.reduce((sum, val) => sum + val, 0);
  return expVals.map((val) => val / sumExpVals);
}

import * as ort from "onnxruntime-web";
import { useEffect, useState } from "react";

export default function Home() {
  const [fen, setFen] = useState("");
  const [output, setOutput] = useState(null);

  async function runModel() {
    try {
      const eloDict = createEloDict();
      const allMovesDict = allPossibleMoves;
      const allMovesDictReversed = createAllMovesDictReversed(allMovesDict);

      // Define a sample FEN and Elo values
      const sampleFEN =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
      const eloSelf = 1100;
      const eloOppo = 1100;

      const result = await inferenceEach(
        "/maia_rapid_onnx.onnx",
        [allMovesDict, eloDict, allMovesDictReversed],
        sampleFEN,
        eloSelf,
        eloOppo
      );
      console.log(result);

      setOutput(result);
    } catch (error) {
      console.error("Error running the model:", error);
    }
  }

  return (
    <div>
      <div>
        <h1>Chess Model Output</h1>
        <input
          type="text"
          value={fen}
          onChange={(e) => setFen(e.target.value)}
          placeholder="Enter FEN"
        />
        <button onClick={runModel}>Run Model</button>
        <pre>{output ? JSON.stringify(output, null, 2) : "Loading..."}</pre>
      </div>
    </div>
  );
}
