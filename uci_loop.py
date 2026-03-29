import sys

import chess

from engine_protocol import ChessEngine
from minimax_search.minimax_search import MinimaxSearchEngine


def uci_loop(engine: ChessEngine) -> None:
    """Runs the Universal Chess Interface for engine for testing."""
    board = chess.Board()

    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        command = line.split()
        cmd_type = command[0]

        if cmd_type == "uci":
            print("id name TestEngine")
            print("id author asdf")
            print("uciok")
            sys.stdout.flush()

        elif cmd_type == "isready":
            print("readyok")
            sys.stdout.flush()

        elif cmd_type == "position":
            if "startpos" in command:
                board.set_fen(chess.STARTING_FEN)
                if "moves" in command:
                    moves_idx = command.index("moves")
                    for move in command[moves_idx + 1 :]:
                        board.push_uci(move)
            elif "fen" in command:
                fen_idx = command.index("fen")
                fen = " ".join(command[fen_idx + 1 : fen_idx + 7])
                board.set_fen(fen)
                if "moves" in command:
                    moves_idx = command.index("moves")
                    for move in command[moves_idx + 1 :]:
                        board.push_uci(move)

        elif cmd_type == "go":
            wtime_s = btime_s = None
            if "wtime" in command:
                wi = command.index("wtime")
                wtime_s = int(command[wi + 1]) / 1000.0
            if "btime" in command:
                bi = command.index("btime")
                btime_s = int(command[bi + 1]) / 1000.0
            remaining = wtime_s if board.turn == chess.WHITE else btime_s
            try:
                best_move = engine.choose_move(board, remaining_time_sec=remaining)
            except TypeError:
                best_move = engine.choose_move(board)
            print(f"bestmove {best_move.uci()}")
            sys.stdout.flush()

        elif cmd_type == "quit":
            break


if __name__ == "__main__":
    engine = MinimaxSearchEngine()
    uci_loop(engine)
