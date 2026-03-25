import random

import chess

from engine_protocol import ChessEngine


class RandomMoveEngine(ChessEngine):
    """Picks a random move from the legal moves."""

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            msg = "no legal moves in position, but game not over?"
            raise ValueError(msg)
        return random.choice(legal)
