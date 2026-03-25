from typing import Protocol

import chess


class ChessEngine(Protocol):
    """Give all engines choose_move method."""

    def choose_move(self, board: chess.Board) -> chess.Move: ...
