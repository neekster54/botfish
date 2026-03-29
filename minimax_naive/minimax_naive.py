import chess

from engine_protocol import ChessEngine

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

MATE_BASE = 1000000  # Big Number


def _material_for_color(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == color:
            total += PIECE_VALUES[piece.piece_type]
    return total


def _evaluate(board: chess.Board) -> int:
    """Basic material evaluator."""
    white = _material_for_color(board, chess.WHITE)
    black = _material_for_color(board, chess.BLACK)
    if board.turn == chess.WHITE:
        return white - black
    return black - white


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    legal = list(board.legal_moves)
    if not legal:
        if board.is_checkmate():
            return -(MATE_BASE + depth)
        return 0
    if depth == 0:
        return _evaluate(board)

    value = -(10**9)
    for move in legal:
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        value = max(value, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return value


class MinimaxEngine(ChessEngine):
    """Depth-limited negamax with alpha-beta and material + mate scoring."""

    # Limited to depth 2 to speed up the evaluation
    def __init__(self, search_depth: int = 2) -> None:
        self.search_depth = search_depth

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            msg = "already game over?"
            raise ValueError(msg)

        # Initial values for minimax
        best_move = legal[0]
        best_value = -(10**9)
        alpha = -(10**9)
        beta = 10**9

        for move in legal:
            board.push(move)
            score = -_negamax(board, self.search_depth - 1, -beta, -alpha)
            board.pop()
            if score > best_value:
                best_value = score
                best_move = move
            alpha = max(alpha, score)
        return best_move
