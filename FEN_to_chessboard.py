import chess
import numpy as np
from encoding_tools import encode_fen_string
from chess_board import board

class FenToChessBoard:
    @staticmethod
    def fen_to_board(fen_str):
        """ 将 FEN 格式转换为 chess_board.board 对象 """
        chess_board = board()
        board_matrix = FenToChessBoard.fen_to_matrix(fen_str)
        chess_board.current_board = board_matrix
        return chess_board

    @staticmethod
    def fen_to_matrix(fen_str):
        """ 将 FEN 格式转换为 8x8 棋盘矩阵 """
        board = chess.Board(fen=fen_str)
        board_matrix = np.full((8, 8), " ", dtype=str)  # 先填充空格

        piece_map = board.piece_map()  # 获取棋盘上的棋子
        for pos, piece in piece_map.items():
            row, col = divmod(pos, 8)  # 坐标转换
            board_matrix[row, col] = piece.symbol()  # 获取棋子符号

        return board_matrix

# 示例用法：
if __name__ == "__main__":
    fen_example = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    converted_board = FenToChessBoard.fen_to_board(fen_example)
    print(converted_board.current_board)
