import chess
import numpy as np

STEP_ENUM = "rnbqkpRNBQKP."

# Defining Data preparation functions
def one_hot_encode_piece(piece):
    pieces = list(STEP_ENUM)
    arr = np.zeros(len(pieces))
    piece_to_index = {p: i for i, p in enumerate(pieces)}
    index = piece_to_index[piece]
    arr[index] = 1
    return arr

def encode_board(board):
    # first lets turn the board into a string
    board_str = str(board)
    # then lets remove all the spaces
    board_str = board_str.replace(' ', '')
    board_list = []
    for row in board_str.split('\n'):
        row_list = []
        for piece in row:
            row_list.append(one_hot_encode_piece(piece))
        board_list.append(row_list)
    return np.array(board_list)

def encode_fen_string(fen_str):
    board = chess.Board(fen=fen_str)
    return encode_board(board)

