import { Chess, Square } from "chess.js";

// Generate all pawn promotions (helper function)
function generatePawnPromotions(): string[] {
  const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const promotions = ["q", "r", "b", "n"];
  const pawnPromotions: string[] = [];

  files.forEach((file) => {
    promotions.forEach((promo) => {
      // Promotion moves for white pawns
      pawnPromotions.push(`${file}7${file}8${promo}`);
      // Promotion moves for black pawns
      pawnPromotions.push(`${file}2${file}1${promo}`);
    });
  });

  return pawnPromotions;
}

// Generate all possible moves in UCI format
function getAllPossibleMoves(): string[] {
  const allMoves: string[] = [];
  const files = "abcdefgh";
  const ranks = "12345678";

  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const square = files[file] + ranks[rank];

      // Generate moves for a queen on the square
      const boardQueen = new Chess();
      boardQueen.clear(); // Clear the board
      boardQueen.put({ type: "q", color: "w" }, square as Square); // Place a white queen
      const queenMoves = boardQueen.moves({
        square: square as Square,
        verbose: true,
      });
      allMoves.push(...queenMoves.map((move) => move.from + move.to));

      // Generate moves for a knight on the square
      const boardKnight = new Chess();
      boardKnight.clear(); // Clear the board
      boardKnight.put({ type: "n", color: "w" }, square as Square); // Place a white knight
      const knightMoves = boardKnight.moves({
        square: square as Square,
        verbose: true,
      });
      allMoves.push(...knightMoves.map((move) => move.from + move.to));
    }
  }

  // Add pawn promotions
  const pawnPromotions = generatePawnPromotions();
  allMoves.push(...pawnPromotions);

  return Array.from(new Set(allMoves)); // Ensure moves are unique
}

// Create a dictionary of all moves and their indices
function createAllMovesDict(): Record<string, number> {
  const allMoves = getAllPossibleMoves();
  const allMovesDict: Record<string, number> = {};
  allMoves.forEach((move, index) => {
    allMovesDict[move] = index;
  });
  return allMovesDict;
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

export { createAllMovesDict, createAllMovesDictReversed, createEloDict };
