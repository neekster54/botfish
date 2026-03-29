import time
from typing import NamedTuple

import chess
from chess.polyglot import zobrist_hash

from engine_protocol import ChessEngine

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # Avoids trying to 'capture' the king
}

# Initial scores for positions
NEG_INF = -(10**9)
POS_INF = 10**9

# Transposition table bounds
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Periodic node count between clock checks (when a deadline is set)
CLOCK_CHECK_INTERVAL = 4096


class _SearchTimeout(Exception):
    """Raised when the search must abort due to time."""


class TTEntry(NamedTuple):
    """Transposition table entry."""

    depth: int
    score: int
    bound: int
    best_move: chess.Move | None = None  # Cache move for hash move ordering


def _material_for_color(board: chess.Board, color: chess.Color) -> int:
    """Count the material for a given color."""
    total = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == color:
            total += PIECE_VALUES[piece.piece_type]
    return total


def _evaluate(board: chess.Board) -> int:
    """Simple material evaluator."""
    white = _material_for_color(board, chess.WHITE)
    black = _material_for_color(board, chess.BLACK)
    if board.turn == chess.WHITE:
        return white - black
    return black - white


def _mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """Try to capture the most valuable piece using least valuable attacker."""
    if not board.is_capture(move):
        return -1
    attacker = board.piece_at(move.from_square)
    if attacker is None:
        return -1
    if board.is_en_passant(move):
        victim_val = PIECE_VALUES[chess.PAWN]
    else:
        victim = board.piece_at(move.to_square)
        if victim is None:
            return -1
        victim_val = PIECE_VALUES[victim.piece_type]
    attacker_val = PIECE_VALUES[attacker.piece_type]
    return victim_val * 100 - attacker_val  # Might need to tweak this


def _order_moves(
    board: chess.Board,
    legal: list[chess.Move],
    tt_move: chess.Move | None,
    pv_move: chess.Move | None,
) -> list[chess.Move]:
    """Add TT move and PV move first, then captures (MVV-LVA), then quiet moves."""
    legal_set = set(legal)
    ordered: list[chess.Move] = []  # Stores the ordered moves
    seen: set[chess.Move] = set()  # Avoids adding the same move twice
    for candidate in (tt_move, pv_move):
        if candidate is not None and candidate in legal_set and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    captures: list[chess.Move] = []  # Moves that capture something (usually good)
    quiet: list[
        chess.Move
    ] = []  # Moves that don't capture anything (usually not as impactful)
    for m in legal:
        if m in seen:
            continue
        if board.is_capture(m):
            captures.append(m)
        else:
            quiet.append(m)
    captures.sort(
        key=lambda mv: _mvv_lva_score(board, mv), reverse=True
    )  # Sort captures by MVV-LVA score
    ordered.extend(captures)
    ordered.extend(quiet)
    return ordered


def _tt_probe(
    tt: dict[int, TTEntry], key: int, depth: int, alpha: int, beta: int
) -> int | None:
    """Return cached score if the entry is usable, otherwise None."""
    entry = tt.get(key)
    if entry is None or entry.depth < depth:
        return None
    if entry.bound == TT_EXACT:
        return entry.score
    if entry.bound == TT_LOWER and entry.score >= beta:
        return entry.score
    if entry.bound == TT_UPPER and entry.score <= alpha:
        return entry.score
    return None


def _tt_store(
    tt: dict[int, TTEntry],
    key: int,
    depth: int,
    score: int,
    beta: int,
    alpha_orig: int,
    best_move: chess.Move | None,
) -> None:
    """Store the minimax result in the transposition table."""
    if score <= alpha_orig:
        bound = TT_UPPER
    elif score >= beta:
        bound = TT_LOWER
    else:
        bound = TT_EXACT
    prev = tt.get(key)
    if prev is not None and depth < prev.depth:
        return
    tt[key] = TTEntry(depth=depth, score=score, bound=bound, best_move=best_move)


def _maybe_abort(deadline: float | None, nodes: list[int]) -> None:
    """Check if the search should be aborted due to time every 4096 branches."""
    if deadline is None:
        return
    nodes[0] += 1
    if (nodes[0] & (CLOCK_CHECK_INTERVAL - 1)) == 0 and time.perf_counter() >= deadline:
        raise _SearchTimeout


def _extract_pv_from_tt(
    board: chess.Board, tt: dict[int, TTEntry], max_plies: int
) -> tuple[chess.Move, ...]:
    """Actually build the PV by following TT best moves."""
    out: list[chess.Move] = []
    bb = board.copy()
    for _ in range(max_plies):
        if bb.is_game_over():
            break
        ent = tt.get(zobrist_hash(bb))
        if ent is None or ent.best_move is None:
            break
        if ent.best_move not in bb.legal_moves:
            break
        out.append(ent.best_move)
        bb.push(ent.best_move)
    return tuple(out)


def _negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    tt: dict[int, TTEntry],
    deadline: float | None,
    nodes: list[int],
    pv_suffix: tuple[chess.Move, ...],
) -> int:
    """Perform a negamax search with alpha-beta pruning and a transposition table."""
    _maybe_abort(deadline, nodes)

    key = zobrist_hash(board)
    cached = _tt_probe(tt, key, depth, alpha, beta)
    if cached is not None:
        return cached

    legal = list(board.legal_moves)
    if not legal:
        if board.is_checkmate():
            score = -(POS_INF + depth)
        else:
            score = 0
        _tt_store(tt, key, depth, score, beta, alpha, None)
        return score
    if depth == 0:
        score = _evaluate(board)
        _tt_store(tt, key, depth, score, beta, alpha, None)
        return score

    # Get the TT move and PV (best current line) moves for move ordering
    entry = tt.get(key)
    tt_move = entry.best_move if entry else None
    pv_move = pv_suffix[0] if pv_suffix else None

    # Order moves to allow for better alpha-beta pruning
    ordered = _order_moves(board, legal, tt_move, pv_move)

    alpha_orig = alpha
    value = NEG_INF
    best_m: chess.Move | None = None
    for move in ordered:
        _maybe_abort(deadline, nodes)
        board.push(move)
        next_pv = pv_suffix[1:] if pv_suffix and move == pv_suffix[0] else ()
        score = -_negamax(board, depth - 1, -beta, -alpha, tt, deadline, nodes, next_pv)
        board.pop()
        if score > value:
            value = score
            best_m = move
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    _tt_store(tt, key, depth, value, beta, alpha_orig, best_m)
    return value


def _search_root(
    board: chess.Board,
    depth: int,
    tt: dict[int, TTEntry],
    deadline: float | None,
    nodes: list[int],
    pv_suffix: tuple[chess.Move, ...],
) -> tuple[chess.Move, int]:
    legal = list(board.legal_moves)
    key = zobrist_hash(board)

    # Get the hashed move
    entry = tt.get(key)
    tt_move = entry.best_move if entry else None

    # Get the PV move
    pv_move = pv_suffix[0] if pv_suffix else None

    # Order moves to allow for better alpha-beta pruning
    ordered = _order_moves(board, legal, tt_move, pv_move)

    # Initialize best move and value to ordered first move (prune better)
    best_move = ordered[0]
    best_value = NEG_INF
    alpha = NEG_INF
    beta = POS_INF

    for move in ordered:
        _maybe_abort(deadline, nodes)
        board.push(move)
        next_pv = pv_suffix[1:] if pv_suffix and move == pv_suffix[0] else ()
        score = -_negamax(board, depth - 1, -beta, -alpha, tt, deadline, nodes, next_pv)
        board.pop()
        if score > best_value:
            best_value = score
            best_move = move
        alpha = max(alpha, score)

    # Root stores best move at this position for future TT ordering / PV walk
    _tt_store(
        tt,
        key,
        depth,
        best_value,
        beta,
        NEG_INF,
        best_move,
    )
    return best_move, best_value


def _time_budget_sec(remaining_time_sec: float | None) -> float | None:
    """Allocate 0.25s per move, else remaining_time/20."""
    if remaining_time_sec is None:
        return None
    return min(0.25, max(0.0, remaining_time_sec) / 20.0)


class MinimaxSearchEngine(ChessEngine):
    """Negamax with TT, MVV-LVA and hash/PV move ordering, iterative deepening."""

    def __init__(self, search_depth: int = 50) -> None:
        self.search_depth = (
            search_depth  # Set to big value since will abort early if time runs out
        )
        self.tt: dict[int, TTEntry] = {}

    def choose_move(
        self, board: chess.Board, *, remaining_time_sec: float | None = None
    ) -> chess.Move:
        """Initialise the iterative deepening search."""
        legal = list(board.legal_moves)
        if not legal:
            msg = "already game over?"
            raise ValueError(msg)

        tt = self.tt

        # Allocate time depending on remaining time
        budget = _time_budget_sec(remaining_time_sec)
        deadline = None if budget is None else time.perf_counter() + budget

        best_move = legal[0]
        pv: tuple[chess.Move, ...] = ()

        # Perform iterative deepening search
        for depth in range(1, self.search_depth + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            nodes: list[int] = [0]
            try:
                best_move, _ = _search_root(board, depth, tt, deadline, nodes, pv)
            except _SearchTimeout:
                break
            pv = _extract_pv_from_tt(board, tt, max_plies=64)

        return best_move
