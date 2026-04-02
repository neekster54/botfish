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

# fmt: off
# Piece-square tables from white's perspective for opening/midgame
_PST_PAWN_MG: tuple[int, ...] = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
)
_PST_KNIGHT_MG: tuple[int, ...] = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)
_PST_BISHOP_MG: tuple[int, ...] = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)
_PST_ROOK_MG: tuple[int, ...] = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
)
_PST_QUEEN_MG: tuple[int, ...] = (
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
)
_PST_KING_MG: tuple[int, ...] = (
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
)
# Different for endgame
_PST_KING_EG: tuple[int, ...] = (
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
)
# fmt: on


_PST_BY_TYPE_MG: dict[chess.PieceType, tuple[int, ...]] = {
    chess.PAWN: _PST_PAWN_MG,
    chess.KNIGHT: _PST_KNIGHT_MG,
    chess.BISHOP: _PST_BISHOP_MG,
    chess.ROOK: _PST_ROOK_MG,
    chess.QUEEN: _PST_QUEEN_MG,
}

# Endgame: Force enemy to corner, encourage moving King closer to opponent's King
_KING_TROPISM_WEIGHT = 12
_ENEMY_CORNER_WEIGHT = 8
_ENDGAME_PHASE_MAX = 256

# Initial scores for positions
NEG_INF = -(10**9)
POS_INF = 10**9

# Transposition table bounds
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Periodic node count between clock checks (when a deadline is set)
CLOCK_CHECK_INTERVAL = 4096

# How many captures deep to search, 8 should be enough
QUIESCENCE_MAX_DEPTH = 8


class _SearchTimeout(Exception):
    """Raised when the search must abort due to time."""


class TTEntry(NamedTuple):
    """Transposition table entry."""

    depth: int
    score: int
    bound: int
    best_move: chess.Move | None = None  # Cache move for hash move ordering


def _pst_for_square(piece: chess.Piece, sq: chess.Square, king_eg_blend: float) -> int:
    """Calculate the piece square table values, handle King separately for midgame/endgame."""
    if piece.piece_type == chess.KING:
        mg = (
            _PST_KING_MG[sq]
            if piece.color == chess.WHITE
            else _PST_KING_MG[chess.square_mirror(sq)]
        )
        eg = (
            _PST_KING_EG[sq]
            if piece.color == chess.WHITE
            else _PST_KING_EG[chess.square_mirror(sq)]
        )
        return int(mg * (1.0 - king_eg_blend) + eg * king_eg_blend)
    table = _PST_BY_TYPE_MG[piece.piece_type]
    idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
    return table[idx]


def _phase_for_board(board: chess.Board) -> int:
    """Calculate game stage where 256 = opening, 0 = pure endgame."""
    phase = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.piece_type == chess.KING:
            continue
        if p.piece_type == chess.PAWN:
            phase += 0
        elif p.piece_type == chess.KNIGHT or p.piece_type == chess.BISHOP:
            phase += 1
        elif p.piece_type == chess.ROOK:
            phase += 2
        elif p.piece_type == chess.QUEEN:
            phase += 4
    return min(_ENDGAME_PHASE_MAX, phase * 32)


def _is_endgame_heuristic(board: chess.Board) -> bool:
    return _phase_for_board(board) <= 96


def _king_square(board: chess.Board, color: chess.Color) -> chess.Square | None:
    k = board.king(color)
    return k


def _enemy_corner_centrality(sq: chess.Square) -> int:
    """Higher when the king is central, lower/better near edges/corners."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    return min(f, 7 - f) + min(r, 7 - r)


def _material_for_color(board: chess.Board, color: chess.Color) -> int:
    """Count the material for a given color."""
    total = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == color:
            total += PIECE_VALUES[piece.piece_type]
    return total


def _pst_total_for_color(
    board: chess.Board, color: chess.Color, king_eg_blend: float
) -> int:
    """Sum the piece square table values for all pieces of a given color."""
    s = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None and p.color == color:
            s += _pst_for_square(p, sq, king_eg_blend)
    return s


def _endgame_king_terms(board: chess.Board, for_color: chess.Color) -> int:
    """Bringn your King closer to enemy king, push towards corners/edges."""
    if not _is_endgame_heuristic(board):
        return 0
    our = _king_square(board, for_color)
    opp = _king_square(board, not for_color)
    if our is None or opp is None:
        return 0
    dist = chess.square_distance(our, opp)
    tropism = (7 - dist) * _KING_TROPISM_WEIGHT
    corner = (6 - _enemy_corner_centrality(opp)) * _ENEMY_CORNER_WEIGHT
    return tropism + corner


def _evaluate(board: chess.Board) -> int:
    """Material + piece-square tables + simple endgame king heuristics to evaluate position."""
    phase = _phase_for_board(board)
    king_eg = (256 - phase) / 256.0

    w_mat = _material_for_color(board, chess.WHITE)
    b_mat = _material_for_color(board, chess.BLACK)
    w_pst = _pst_total_for_color(board, chess.WHITE, king_eg)
    b_pst = _pst_total_for_color(board, chess.BLACK, king_eg)

    w_eg = _endgame_king_terms(board, chess.WHITE)
    b_eg = _endgame_king_terms(board, chess.BLACK)

    white_total = w_mat + w_pst + w_eg
    black_total = b_mat + b_pst + b_eg

    if board.turn == chess.WHITE:
        return white_total - black_total
    return black_total - white_total


def _victim_value(board: chess.Board, move: chess.Move) -> int:
    """Value of piece being captured by move, or 0 if not a capture."""
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    victim = board.piece_at(move.to_square)
    if victim is None:
        return 0
    return PIECE_VALUES[victim.piece_type]


def _least_valuable_attacker(
    board: chess.Board, target_sq: chess.Square
) -> chess.Move | None:
    """Legal capture onto target_sq by side to move, smallest attacker first."""
    side = board.turn
    best: chess.Move | None = None
    best_val = POS_INF
    for mv in board.legal_moves:
        if mv.to_square != target_sq or not board.is_capture(mv):
            continue
        p = board.piece_at(mv.from_square)
        if p is None or p.color != side:
            continue
        v = PIECE_VALUES[p.piece_type]
        if v < best_val:
            best_val = v
            best = mv
    return best


def _see_swap(board: chess.Board, sq: chess.Square) -> int:
    """Calculate gain for side to move if a piece on square is captured."""
    mv = _least_valuable_attacker(board, sq)
    if mv is None:
        return 0
    victim = board.piece_at(sq)
    if victim is None:
        return 0
    gain = PIECE_VALUES[victim.piece_type]
    board.push(mv)
    rest = _see_swap(board, sq)
    board.pop()
    return gain - rest


def _static_exchange_eval(board: chess.Board, move: chess.Move) -> int:
    """Calculate the static exchange evaluation (SEE) for a capture move to find net material gain/loss."""
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    if not board.is_capture(move):
        return 0

    b = board.copy()
    sq = move.to_square
    gain0 = _victim_value(b, move)
    if move not in b.legal_moves:
        return 0
    b.push(move)
    return gain0 - _see_swap(b, sq)


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


def _is_tactical(board: chess.Board, move: chess.Move) -> bool:
    """Captures and promotions need to calculate deeper."""
    return board.is_capture(move) or move.promotion is not None


def _quiescence(
    board: chess.Board,
    alpha: int,
    beta: int,
    depth: int,
    deadline: float | None,
    nodes: list[int],
) -> int:
    """Avoid Horizon effect by searching captures/promotions until quiet position or max depth."""
    _maybe_abort(deadline, nodes)

    in_check = board.is_check()
    if not in_check:
        stand = _evaluate(board)
        if stand >= beta:
            return beta
        alpha = max(alpha, stand)

    if depth <= 0:
        return alpha

    if in_check:
        tactical = list(board.legal_moves)
    else:
        tactical = [mv for mv in board.legal_moves if _is_tactical(board, mv)]
    if not tactical:
        if in_check:
            return -(POS_INF + depth)
        return alpha
    tactical.sort(key=lambda m: _mvv_lva_score(board, m), reverse=True)

    for move in tactical:
        if not in_check and board.is_capture(move):
            if _static_exchange_eval(board, move) < 0:
                continue
        _maybe_abort(deadline, nodes)
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, depth - 1, deadline, nodes)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha


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


def _move_creates_threefold(board: chess.Board, move: chess.Move) -> bool:
    """Check if making the move would create a threefold repetition."""
    b = board.copy()
    b.push(move)
    return b.is_repetition(3)


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
        score = _quiescence(board, alpha, beta, QUIESCENCE_MAX_DEPTH, deadline, nodes)
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
) -> tuple[chess.Move, int, list[tuple[chess.Move, int]]]:
    legal = list(board.legal_moves)
    key = zobrist_hash(board)

    # Get the hashed move
    entry = tt.get(key)
    tt_move = entry.best_move if entry else None

    # Get the PV move
    pv_move = pv_suffix[0] if pv_suffix else None

    # Order moves to allow for better alpha-beta pruning
    ordered = _order_moves(board, legal, tt_move, pv_move)

    root_scores: list[tuple[chess.Move, int]] = []
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
        root_scores.append((move, score))
        if score > best_value:
            best_value = score
            best_move = move
        alpha = max(alpha, score)

    # Prefer a non-threefold move when another move scores equally well.
    max_s = max(s for _, s in root_scores)
    ties = [m for m, s in root_scores if s == max_s]
    non_rep = [m for m in ties if not _move_creates_threefold(board, m)]
    if non_rep:
        best_move = non_rep[0]

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
    return best_move, best_value, root_scores


def _time_budget_sec(remaining_time_sec: float | None) -> float | None:
    """Allocate 0.25s per move, else remaining_time/20."""
    if remaining_time_sec is None:
        return None
    return min(0.25, max(0.0, remaining_time_sec) / 20.0)


class MinimaxSearchEngine(ChessEngine):
    """Negamax with TT, MVV-LVA, PST, quiescence (SEE), endgame king terms, iterative deepening."""

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
                best_move, _, _ = _search_root(board, depth, tt, deadline, nodes, pv)
            except _SearchTimeout:
                break
            pv = _extract_pv_from_tt(board, tt, max_plies=64)

        return best_move
