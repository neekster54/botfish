import chess

from engine_protocol import ChessEngine
from minimax_endgame.minimax_endgame import MinimaxSearchEngine


def _prompt_user_move(board: chess.Board) -> chess.Move:
    while True:
        raw = input("Your move (UCI, e.g. e2e4): ").strip()
        if not raw:
            continue
        if raw == "quit":
            return None
        try:
            move = chess.Move.from_uci(raw)
        except chess.InvalidMoveError:
            print("Invalid UCI. Use long algebraic form (e2e4, e7e8q for promotion).")
            continue
        if move not in board.legal_moves:
            print("Illegal move for this position. Try again.")
            continue
        return move


def main() -> None:
    board = chess.Board()
    engine: ChessEngine = MinimaxSearchEngine()
    user_color = chess.WHITE

    print("Test: You play White, the engine plays Black. Moves in UCI.")
    print(board)

    while not board.is_game_over():
        if board.turn == user_color:
            move = _prompt_user_move(board)
            if move is None:
                print("Game terminated by user.")
                return
            board.push(move)
        else:
            move = engine.choose_move(board, remaining_time_sec=1.0)
            print(f"Engine: {move.uci()}")
            board.push(move)

        print(board)

    outcome = board.outcome()
    if outcome is None:
        print("Game over (no outcome recorded)?")
        return
    if outcome.winner is None:
        print("Draw.", outcome.termination)
    elif outcome.winner == user_color:
        print("You win —", outcome.termination)
    else:
        print("Engine wins —", outcome.termination)


if __name__ == "__main__":
    main()
