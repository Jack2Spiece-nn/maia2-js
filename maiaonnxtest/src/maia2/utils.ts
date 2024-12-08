import { Chess } from "chess.js";

import allPossibleMovesDict from "./all_moves.json";
import allPossibleMovesReversedDict from "./all_moves_reversed.json";

const allPossibleMoves = allPossibleMovesDict as Record<string, number>;
const allPossibleMovesReversed = allPossibleMovesReversedDict as Record<
  number,
  string
>;

const eloDict = createEloDict();

function softmax(values: Float32Array): Float32Array {
  const maxVal = Math.max(...values);
  const expVals = values.map((val) => Math.exp(val - maxVal));
  const sumExpVals = expVals.reduce((sum, val) => sum + val, 0);
  return expVals.map((val) => val / sumExpVals);
}

/**
 * Converts a chess board state into a tensor representation.
 *
 * The tensor has the following structure:
 * - 12 channels for piece types (6 types for each color)
 * - 1 channel for the player's turn
 * - 4 channels for castling rights
 * - 1 channel for en passant target square
 *
 * @param {Chess} board - The chess board state.
 * @returns {Float32Array} - The tensor representation of the board state.
 */
function boardToTensor(board: Chess): Float32Array {
  const pieceTypes = ["p", "n", "b", "r", "q", "k"];
  const numPieceChannels = 12; // 6 piece types * 2 colors
  const additionalChannels = 6; // Turn, castling rights, en passant
  const tensor = new Float32Array(
    (numPieceChannels + additionalChannels) * 8 * 8
  );

  const pieceIndices: Record<string, number> = pieceTypes.reduce(
    (acc: Record<string, number>, piece, idx) => {
      acc[piece] = idx;
      return acc;
    },
    {} as Record<string, number>
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
  eloOppo: number
): {
  boardInput: Float32Array;
  eloSelf: number;
  eloOppo: number;
  legalMoves: Float32Array;
} {
  // Handle mirroring if it's black's turn
  let board = new Chess(fen);
  if (fen.split(" ")[1] === "b") {
    board = new Chess(mirrorBoard(board.fen()));
  } else if (fen.split(" ")[1] !== "w") {
    throw new Error(`Invalid FEN: ${fen}`);
  }

  // Convert board to tensor
  const boardInput = boardToTensor(board);

  // Map Elo to categories
  const eloSelfCategory = mapToCategory(eloSelf, eloDict);
  const eloOppoCategory = mapToCategory(eloOppo, eloDict);

  // Generate legal moves tensor
  const legalMoves = new Float32Array(Object.keys(allPossibleMoves).length);
  for (const move of board.moves({ verbose: true })) {
    const moveIndex = allPossibleMoves[move.lan];

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

// Create a reversed dictionary of indices to moves
function createAllMovesDictReversed(
  allMovesDict: Record<string, number>
): Record<number, string> {
  const reversed: Record<number, string> = {};
  for (const [move, index] of Object.entries(allMovesDict)) {
    reversed[index] = move;
  }
  return reversed;
}

function createEloDict(): { [key: string]: number } {
  const interval = 100;
  const start = 1100;
  const end = 2000;

  const eloDict: { [key: string]: number } = { [`<${start}`]: 0 };
  let rangeIndex = 1;

  for (let lowerBound = start; lowerBound < end; lowerBound += interval) {
    const upperBound = lowerBound + interval;
    eloDict[`${lowerBound}-${upperBound - 1}`] = rangeIndex;
    rangeIndex += 1;
  }

  eloDict[`>=${end}`] = rangeIndex;

  return eloDict;
}

// Mirror moves for black pieces
function mirrorMove(move: string): string {
  const files = "abcdefgh";
  const ranks = "12345678";
  const [fromFile, fromRank, toFile, toRank] = move;
  const mirroredFrom =
    files[7 - files.indexOf(fromFile)] + ranks[8 - parseInt(fromRank)];
  const mirroredTo =
    files[7 - files.indexOf(toFile)] + ranks[8 - parseInt(toRank)];
  return mirroredFrom + mirroredTo;
}

/**
 * Mirrors a given FEN string to represent the board from the opposite perspective.
 *
 * This function flips the board, swaps the piece colors, and adjusts the turn, castling rights,
 * and en passant target square accordingly.
 *
 * @param {string} fen - The FEN string representing the current board state.
 * @returns {string} - The mirrored FEN string representing the board from the opposite perspective.
 */
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

export {
  preprocess,
  boardToTensor,
  createAllMovesDictReversed,
  eloDict,
  mirrorMove,
  mirrorBoard,
  allPossibleMoves,
  allPossibleMovesReversed,
  mapToCategory,
  softmax,
};
