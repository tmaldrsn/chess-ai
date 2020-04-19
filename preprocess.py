import chess
import chess.pgn
import numpy as np


def read_pgn(filename, color=chess.WHITE):
    pgn = open(filename)

    games = []
    while True:
        game = chess.pgn.read_game(pgn)
        if game.headers['Termination'] == 'Normal' and len(list(game.mainline())) > 20:
            if game.headers['Result'] == '1-0' and color:
                games.append(game)
            elif game.headers['Result'] == '0-1' and not color:
                games.append(game)




def get_white_piece_locations(board):
    return '{:064b}'.format(int(chess.SquareSet([i for i in range(64) if str(board.piece_at(i)).isupper()])))

def get_black_piece_locations(board):
    return '{:064b}'.format(int(chess.SquareSet([i for i in range(64) if str(board.piece_at(i)).islower()])))

def get_all_piece_locations(board):
    return '{:064b}'.format(int(chess.SquareSet([i for i in range(64) if board.piece_at(i)])))


def get_data(game, lookback=1):
    mainline = game.mainline()
    only_pieces = [get_all_piece_locations(move.board()) for move in mainline]
    X = np.array([list(map(int, ''.join(only_pieces[i:i+lookback]))) for i in range(len(only_pieces)-lookback)])
    y = np.array([list(map(int, i)) for i in only_pieces[lookback:]])
    return X, y

def get_piece_location_array(game, move):
    return np.fliplr(np.array(list(get_all_piece_locations(list(game.mainline())[move].board())), dtype=np.int).reshape(8, 8))

def square_from_number(x):
    FILES = "ABCDEFGH"
    RANKS = "87654321"
    return f"{FILES[x % 8]}{RANKS[x // 8]}"

def squares_from_list(l):
    return [square_from_number(x) for x in l]