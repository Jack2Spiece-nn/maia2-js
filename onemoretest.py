import onnx
import torch
import chess
import numpy as np
import onnxruntime as ort


def map_to_category(elo, elo_dict):

    inteval = 100
    start = 1100
    end = 2000

    if elo < start:
        return elo_dict[f"<{start}"]
    elif elo >= end:
        return elo_dict[f">={end}"]
    else:
        for lower_bound in range(start, end - 1, inteval):
            upper_bound = lower_bound + inteval
            if lower_bound <= elo < upper_bound:
                return elo_dict[f"{lower_bound}-{upper_bound - 1}"]


def generate_pawn_promotions():
    # Define the promotion rows for both colors and the promotion pieces
    # promotion_rows = {'white': '7', 'black': '2'}
    promotion_rows = {"white": "7"}
    promotion_pieces = ["q", "r", "b", "n"]
    promotions = []

    # Iterate over each color
    for color, row in promotion_rows.items():
        # Target rows for promotion (8 for white, 1 for black)
        target_row = "8" if color == "white" else "1"

        # Each file from 'a' to 'h'
        for file in "abcdefgh":
            # Direct move to promotion
            for piece in promotion_pieces:
                promotions.append(f"{file}{row}{file}{target_row}{piece}")

            # Capturing moves to the left and right (if not on the edges of the board)
            if file != "a":
                left_file = chr(ord(file) - 1)  # File to the left
                for piece in promotion_pieces:
                    promotions.append(
                        f"{file}{row}{left_file}{target_row}{piece}"
                    )

            if file != "h":
                right_file = chr(ord(file) + 1)  # File to the right
                for piece in promotion_pieces:
                    promotions.append(
                        f"{file}{row}{right_file}{target_row}{piece}"
                    )

    return promotions


def get_all_possible_moves():

    all_moves = []

    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)

            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)

            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)

    all_moves = [all_moves[i].uci() for i in range(len(all_moves))]

    pawn_promotions = generate_pawn_promotions()

    return all_moves + pawn_promotions


def create_elo_dict():

    inteval = 100
    start = 1100
    end = 2000

    range_dict = {f"<{start}": 0}
    range_index = 1

    for lower_bound in range(start, end - 1, inteval):
        upper_bound = lower_bound + inteval
        range_dict[f"{lower_bound}-{upper_bound - 1}"] = range_index
        range_index += 1

    range_dict[f">={end}"] = range_index

    # print(range_dict, flush=True)

    return range_dict


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )


def mirror_move(move_uci):
    # Check if the move is a promotion (length of UCI string will be more than 4)
    is_promotion = len(move_uci) > 4

    # Extract the start and end squares, and the promotion piece if applicable
    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    promotion_piece = move_uci[4:] if is_promotion else ""

    # Mirror the start and end squares
    mirrored_start = mirror_square(start_square)
    mirrored_end = mirror_square(end_square)

    # Return the mirrored move, including the promotion piece if applicable
    return mirrored_start + mirrored_end + promotion_piece


def mirror_square(square):

    file = square[0]
    rank = str(9 - int(square[1]))

    return file + rank


def board_to_tensor(board):

    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    num_piece_channels = 12  # 6 piece types * 2 colors
    additional_channels = (
        6  # 1 for player's turn, 4 for castling rights, 1 for en passant
    )
    tensor = torch.zeros(
        (num_piece_channels + additional_channels, 8, 8), dtype=torch.float32
    )

    # Precompute indices for each piece type
    piece_indices = {piece: i for i, piece in enumerate(piece_types)}

    # Fill tensor for each piece type
    for piece_type in piece_types:
        for color in [True, False]:  # True is White, False is Black
            piece_map = board.pieces(piece_type, color)
            index = piece_indices[piece_type] + (0 if color else 6)
            for square in piece_map:
                row, col = divmod(square, 8)
                tensor[index, row, col] = 1.0

    # Player's turn channel (White = 1, Black = 0)
    turn_channel = num_piece_channels
    if board.turn == chess.WHITE:
        tensor[turn_channel, :, :] = 1.0

    # Castling rights channels
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    for i, has_right in enumerate(castling_rights):
        if has_right:
            tensor[num_piece_channels + 1 + i, :, :] = 1.0

    # En passant target channel
    ep_channel = num_piece_channels + 5
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[ep_channel, row, col] = 1.0

    return tensor


def preprocessing(fen, elo_self, elo_oppo, elo_dict, all_moves_dict):

    if fen.split(" ")[1] == "w":
        board = chess.Board(fen)
    elif fen.split(" ")[1] == "b":
        board = chess.Board(fen).mirror()
    else:
        raise ValueError(f"Invalid fen: {fen}")

    board_input = board_to_tensor(board)

    elo_self = map_to_category(elo_self, elo_dict)
    elo_oppo = map_to_category(elo_oppo, elo_dict)

    legal_moves = torch.zeros(len(all_moves_dict))
    legal_moves_idx = torch.tensor(
        [all_moves_dict[move.uci()] for move in board.legal_moves]
    )
    legal_moves[legal_moves_idx] = 1

    return board_input, elo_self, elo_oppo, legal_moves


all_moves = get_all_possible_moves()
all_moves_dict = {move: i for i, move in enumerate(all_moves)}
all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
elo_dict = create_elo_dict()

board = chess.Board(fen="8/4pqP1/2Kp1Pk1/2bB3R/n2Pb2p/7N/3R4/8 w - - 0 1")
board_input, elo_self, elo_oppo, legal_moves = preprocessing(
    board.fen(), 1100, 1100, elo_dict, all_moves_dict
)

boards2 = to_numpy(torch.tensor(board_to_tensor(board)).unsqueeze(0))

boards = board_input.unsqueeze(0)
elos_self = to_numpy(torch.tensor([elo_self]))
elos_oppo = to_numpy(torch.tensor([elo_oppo]))

ort_session = ort.InferenceSession("./maia_rapid_onnx.onnx")
ort_inputs = {
    "boards": boards2,
    "elo_self": elos_self,
    "elo_oppo": elos_oppo,
}
ort_outs = ort_session.run(None, ort_inputs)
logits_maia_legal = ort_outs[0] * legal_moves.numpy()
probs = torch.tensor(logits_maia_legal).softmax(dim=-1).cpu().tolist()
preds = np.argmax(logits_maia_legal, axis=-1)


black_flag = False
if board.fen().split(" ")[1] == "b":
    # logits_value = 1 - logits_value
    black_flag = True


move_probs = {}
legal_move_indices = legal_moves.nonzero().flatten().cpu().numpy().tolist()
legal_moves_mirrored = []
for move_idx in legal_move_indices:
    move = all_moves_dict_reversed[move_idx]
    if black_flag:
        move = mirror_move(move)
    legal_moves_mirrored.append(move)

for j in range(len(legal_move_indices)):
    move_probs[legal_moves_mirrored[j]] = round(
        probs[0][legal_move_indices[j]], 4
    )

move_probs = dict(
    sorted(move_probs.items(), key=lambda item: item[1], reverse=True)
)

print(move_probs)
