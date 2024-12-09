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
function boardToTensor(fen: string): Float32Array {
  const tokens = fen.split(" ");
  const piecePlacement = tokens[0];
  const activeColor = tokens[1];
  const castlingAvailability = tokens[2];
  const enPassantTarget = tokens[3];

  const pieceTypes = [
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
  ];
  const tensor = new Float32Array((12 + 6) * 8 * 8);

  const rows = piecePlacement.split("/");

  // Adjust rank indexing
  for (let rank = 0; rank < 8; rank++) {
    const row = 7 - rank; // Invert rank to match Python's row indexing
    let file = 0;
    for (const char of rows[rank]) {
      if (isNaN(parseInt(char))) {
        const index = pieceTypes.indexOf(char);
        const tensorIndex = index * 64 + row * 8 + file;
        tensor[tensorIndex] = 1.0;
        file += 1;
      } else {
        file += parseInt(char);
      }
    }
  }

  // Player's turn channel
  const turnChannelStart = 12 * 64;
  const turnChannelEnd = turnChannelStart + 64;
  const turnValue = activeColor === "w" ? 1.0 : 0.0;
  tensor.fill(turnValue, turnChannelStart, turnChannelEnd);

  // Castling rights channels
  const castlingRights = [
    castlingAvailability.includes("K"),
    castlingAvailability.includes("Q"),
    castlingAvailability.includes("k"),
    castlingAvailability.includes("q"),
  ];
  for (let i = 0; i < 4; i++) {
    if (castlingRights[i]) {
      const channelStart = (13 + i) * 64;
      const channelEnd = channelStart + 64;
      tensor.fill(1.0, channelStart, channelEnd);
    }
  }

  // En passant target channel
  const epChannel = 17 * 64;
  if (enPassantTarget !== "-") {
    const file = enPassantTarget.charCodeAt(0) - "a".charCodeAt(0);
    const rank = parseInt(enPassantTarget[1], 10) - 1; // Adjust rank indexing
    const row = 7 - rank; // Invert rank to match tensor indexing
    const index = epChannel + row * 8 + file;
    tensor[index] = 1.0;
  }

  return tensor;
}

function preprocess(
  fen: string,
  eloSelf: number,
  eloOppo: number
): {
  boardInput: Float32Array;
  eloSelfCategory: number;
  eloOppoCategory: number;
  legalMoves: Float32Array;
} {
  // Handle mirroring if it's black's turn
  let board = new Chess(fen);
  if (fen.split(" ")[1] === "b") {
    board = new Chess(mirrorFEN(board.fen()));
    console.log(board.fen());
  } else if (fen.split(" ")[1] !== "w") {
    throw new Error(`Invalid FEN: ${fen}`);
  }

  // Convert board to tensor
  const boardInput = boardToTensor(board.fen());

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
    eloSelfCategory,
    eloOppoCategory,
    legalMoves,
  };
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

function mirrorMove(moveUci: string): string {
  const isPromotion: boolean = moveUci.length > 4;

  const startSquare: string = moveUci.substring(0, 2);
  const endSquare: string = moveUci.substring(2, 4);
  const promotionPiece: string = isPromotion ? moveUci.substring(4) : "";

  const mirroredStart: string = mirrorSquare(startSquare);
  const mirroredEnd: string = mirrorSquare(endSquare);

  return mirroredStart + mirroredEnd + promotionPiece;
}

function mirrorSquare(square: string): string {
  const file: string = square.charAt(0);
  const rank: string = (9 - parseInt(square.charAt(1))).toString();

  return file + rank;
}

/**
 * Mirrors a chess board vertically and swaps piece colors based on a FEN string.
 * @param fen The original FEN string.
 * @returns The mirrored FEN string.
 */
function mirrorFEN(fen: string): string {
  const [position, activeColor, castling, enPassant, halfmove, fullmove] =
    fen.split(" ");

  // Split the position into ranks
  const ranks = position.split("/");

  // Mirror the ranks (top-to-bottom flip)
  const mirroredRanks = ranks
    .slice()
    .reverse()
    .map((rank) => swapColorsInRank(rank));

  // Reconstruct the mirrored position
  const mirroredPosition = mirroredRanks.join("/");

  // Swap active color
  const mirroredActiveColor = activeColor === "w" ? "b" : "w";

  // Adjust castling rights: Swap uppercase (white) with lowercase (black) and vice versa
  const mirroredCastling = swapCastlingRights(castling);

  // En passant square: Mirror the rank only (since flipping top-to-bottom)
  const mirroredEnPassant =
    enPassant !== "-" ? mirrorEnPassant(enPassant) : "-";

  // Return the new FEN
  return `${mirroredPosition} ${mirroredActiveColor} ${mirroredCastling} ${mirroredEnPassant} ${halfmove} ${fullmove}`;
}

/**
 * Swaps the colors of the pieces in a rank.
 * Uppercase letters represent White pieces, lowercase represent Black.
 * @param rank The rank string from FEN.
 * @returns The rank string with swapped piece colors.
 */
function swapColorsInRank(rank: string): string {
  let swappedRank = "";
  for (const char of rank) {
    if (/[A-Z]/.test(char)) {
      swappedRank += char.toLowerCase();
    } else if (/[a-z]/.test(char)) {
      swappedRank += char.toUpperCase();
    } else {
      // Numbers representing empty squares
      swappedRank += char;
    }
  }
  return swappedRank;
}

/**
 * Swaps the castling rights by changing uppercase to lowercase and vice versa.
 * @param castling The castling rights string from FEN.
 * @returns The swapped castling rights string.
 */
function swapCastlingRights(castling: string): string {
  if (castling === "-") return castling;

  let swapped = "";
  for (const char of castling) {
    if (/[A-Z]/.test(char)) {
      swapped += char.toLowerCase();
    } else if (/[a-z]/.test(char)) {
      swapped += char.toUpperCase();
    } else {
      swapped += char;
    }
  }

  return swapped;
}

/**
 * Mirrors the en passant square vertically (top-to-bottom flip).
 * Only the rank is affected; the file remains the same.
 * @param square The en passant square in algebraic notation.
 * @returns The mirrored en passant square.
 */
function mirrorEnPassant(square: string): string {
  const file = square[0];
  const rank = square[1];

  // Flip the rank: '1' ↔ '8', '2' ↔ '7', etc.
  const mirroredRank = (9 - parseInt(rank, 10)).toString();

  return `${file}${mirroredRank}`;
}

export {
  preprocess,
  boardToTensor,
  mirrorMove,
  createAllMovesDictReversed,
  eloDict,
  mirrorFEN,
  allPossibleMoves,
  allPossibleMovesReversed,
  mapToCategory,
  softmax,
};
