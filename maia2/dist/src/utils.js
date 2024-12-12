"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.allPossibleMovesReversed = void 0;
exports.preprocess = preprocess;
exports.mirrorMove = mirrorMove;
const chess_js_1 = require("chess.js");
const all_moves_json_1 = __importDefault(require("./data/all_moves.json"));
const all_moves_reversed_json_1 = __importDefault(require("./data/all_moves_reversed.json"));
const allPossibleMoves = all_moves_json_1.default;
const allPossibleMovesReversed = all_moves_reversed_json_1.default;
exports.allPossibleMovesReversed = allPossibleMovesReversed;
const eloDict = createEloDict();
/**
 * Converts a chess board position in FEN notation to a tensor representation.
 * The tensor includes information about piece placement, active color, castling rights, and en passant target.
 *
 * @param fen - The FEN string representing the chess board position.
 * @returns A Float32Array representing the tensor of the board position.
 */
function boardToTensor(fen) {
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
        const row = 7 - rank;
        let file = 0;
        for (const char of rows[rank]) {
            if (isNaN(parseInt(char))) {
                const index = pieceTypes.indexOf(char);
                const tensorIndex = index * 64 + row * 8 + file;
                tensor[tensorIndex] = 1.0;
                file += 1;
            }
            else {
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
/**
 * Preprocesses the input data for the model.
 * Converts the FEN string, Elo ratings, and legal moves into tensors.
 *
 * @param fen - The FEN string representing the board position.
 * @param eloSelf - The Elo rating of the player.
 * @param eloOppo - The Elo rating of the opponent.
 * @returns An object containing the preprocessed data.
 */
function preprocess(fen, eloSelf, eloOppo) {
    // Handle mirroring if it's black's turn
    let board = new chess_js_1.Chess(fen);
    if (fen.split(" ")[1] === "b") {
        board = new chess_js_1.Chess(mirrorFEN(board.fen()));
    }
    else if (fen.split(" ")[1] !== "w") {
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
/**
 * Maps an Elo rating to a predefined category based on specified intervals.
 *
 * @param elo - The Elo rating to be categorized.
 * @param eloDict - A dictionary mapping Elo ranges to category indices.
 * @returns The category index corresponding to the given Elo rating.
 * @throws Will throw an error if the Elo value is out of the predefined range.
 */
function mapToCategory(elo, eloDict) {
    const interval = 100;
    const start = 1100;
    const end = 2000;
    if (elo < start) {
        return eloDict[`<${start}`];
    }
    else if (elo >= end) {
        return eloDict[`>=${end}`];
    }
    else {
        for (let lowerBound = start; lowerBound < end; lowerBound += interval) {
            const upperBound = lowerBound + interval;
            if (elo >= lowerBound && elo < upperBound) {
                return eloDict[`${lowerBound}-${upperBound - 1}`];
            }
        }
    }
    throw new Error("Elo value is out of range.");
}
/**
 * Creates a dictionary mapping Elo rating ranges to category indices.
 *
 * @returns A dictionary mapping Elo ranges to category indices.
 */
function createEloDict() {
    const interval = 100;
    const start = 1100;
    const end = 2000;
    const eloDict = { [`<${start}`]: 0 };
    let rangeIndex = 1;
    for (let lowerBound = start; lowerBound < end; lowerBound += interval) {
        const upperBound = lowerBound + interval;
        eloDict[`${lowerBound}-${upperBound - 1}`] = rangeIndex;
        rangeIndex += 1;
    }
    eloDict[`>=${end}`] = rangeIndex;
    return eloDict;
}
/**
 * Mirrors a chess move in UCI notation.
 * The move is mirrored vertically (top-to-bottom flip) on the board.
 *
 * @param moveUci - The move to be mirrored in UCI notation.
 * @returns The mirrored move in UCI notation.
 */
function mirrorMove(moveUci) {
    const isPromotion = moveUci.length > 4;
    const startSquare = moveUci.substring(0, 2);
    const endSquare = moveUci.substring(2, 4);
    const promotionPiece = isPromotion ? moveUci.substring(4) : "";
    const mirroredStart = mirrorSquare(startSquare);
    const mirroredEnd = mirrorSquare(endSquare);
    return mirroredStart + mirroredEnd + promotionPiece;
}
/**
 * Mirrors a square on the chess board vertically (top-to-bottom flip).
 * The file remains the same, while the rank is inverted.
 *
 * @param square - The square to be mirrored in algebraic notation.
 * @returns The mirrored square in algebraic notation.
 */
function mirrorSquare(square) {
    const file = square.charAt(0);
    const rank = (9 - parseInt(square.charAt(1))).toString();
    return file + rank;
}
/**
 * Mirrors a FEN string vertically (top-to-bottom flip).
 * The active color, castling rights, and en passant target are adjusted accordingly.
 *
 * @param fen - The FEN string to be mirrored.
 * @returns The mirrored FEN string.
 */
function mirrorFEN(fen) {
    const [position, activeColor, castling, enPassant, halfmove, fullmove] = fen.split(" ");
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
    const mirroredEnPassant = enPassant !== "-" ? mirrorEnPassant(enPassant) : "-";
    // Return the new FEN
    return `${mirroredPosition} ${mirroredActiveColor} ${mirroredCastling} ${mirroredEnPassant} ${halfmove} ${fullmove}`;
}
/**
 * Swaps the colors of pieces in a rank by changing uppercase to lowercase and vice versa.
 * @param rank The rank to be mirrored.
 * @returns The mirrored rank.
 */
function swapColorsInRank(rank) {
    let swappedRank = "";
    for (const char of rank) {
        if (/[A-Z]/.test(char)) {
            swappedRank += char.toLowerCase();
        }
        else if (/[a-z]/.test(char)) {
            swappedRank += char.toUpperCase();
        }
        else {
            // Numbers representing empty squares
            swappedRank += char;
        }
    }
    return swappedRank;
}
/**
 * Swaps the colors of pieces in a rank by changing uppercase to lowercase and vice versa.
 * @param rank The rank to be mirrored.
 * @returns The mirrored rank.
 */
function swapCastlingRights(castling) {
    if (castling === "-")
        return castling;
    let swapped = "";
    for (const char of castling) {
        if (/[A-Z]/.test(char)) {
            swapped += char.toLowerCase();
        }
        else if (/[a-z]/.test(char)) {
            swapped += char.toUpperCase();
        }
        else {
            swapped += char;
        }
    }
    return swapped;
}
/**
 * Mirrors the en passant target square by flipping the rank.
 * @param square The en passant target square to be mirrored.
 * @returns The mirrored en passant target square.
 */
function mirrorEnPassant(square) {
    const file = square[0];
    const rank = square[1];
    // Flip the rank: '1' ↔ '8', '2' ↔ '7', etc.
    const mirroredRank = (9 - parseInt(rank, 10)).toString();
    return `${file}${mirroredRank}`;
}
