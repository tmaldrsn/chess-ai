import datetime
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.neural_network import MLPRegressor
import joblib
import chess

import preprocess as pre

N_GAMES = 1000
TRAIN_SPLIT = int(0.8*N_GAMES)
LOOKBACK = 5

class Board(chess.Board):
    def __init__(self, *args, **kwargs):
        chess.Board.__init__(self, *args, **kwargs)
    
    @property
    def binary_board(self):
        return np.flipud(self._binary_board())        
        
    def _binary_board(self):
        return np.array([1 if self.piece_at(i) else 0 for i in range(64)]).reshape(8, 8)



def load_model(filename):
    return joblib.load(filename)

class ChessAgent(MLPRegressor):
    def __init__(self, *args, **kwargs):
        MLPRegressor.__init__(self, *args, **kwargs)

    def train_from_game(self, game):
        """Train the model from one game instance."""      
        self.partial_fit(*pre.get_data(game, 1))    # TODO: Fix this, lookback should be removed

    def train_from_pgn(self, filename, n_games=None):
        """Further train the model using games from a pgn file"""
        pgn = open(filename)
        i = 1
        
        t_games = np.inf if not n_games else n_games
        while True:
            try:
                game = chess.pgn.read_game(pgn)
                if not game:
                    break
                #if game.headers['Termination'] != 'Normal' or len(list(game.mainline())) < 20:
                if game.headers['Termination'] != 'Normal' or len(list(game.mainline())) < 4:
                    if t_games:
                        t_games -= 1
                    continue
                self.train_from_game(game)
                if not n_games:
                    print(f"{i+1} games parsed.", end='\r')
                else:
                    print(f"{i+1} out of {t_games} games parsed. Training {(i+1)/t_games:.2%} complete.", end='\r')
                i += 1
                if i > t_games:
                    break
            except KeyboardInterrupt:
                break
        print(f"Model Training Complete: {i} games trained on.")


    def predict_(self, board):
        array = board._binary_board().flatten().reshape(1, -1)
        return self.predict(array).reshape(8, 8)

    def loss_array(self, board):
        return self.predict_(board) - board._binary_board()

    def best_move(self, board, verbose=True):
        legal_moves = list(board.generate_legal_moves())
        best_move, high_score = chess.Move.null(), -np.inf
        for move in legal_moves:
            t = move.to_square
            f = move.from_square

            t_val = self.loss_array(board).flatten()[t]
            f_val = self.loss_array(board).flatten()[f]

            score = t_val-f_val
            if score > high_score:
                best_move = move
                high_score = score

            if verbose:
                print(f"{move} -> {round(score, 4)}")

        return best_move, high_score


    def save_model(self):
        """Save the model as a pickle in a file"""
        filename = 'models/' + datetime.datetime.now().replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S") + '.pkl'
        joblib.dump(self, filename)
        print(f"Model has been saved: {filename}")


def play_game(opp, color=chess.WHITE, verbose=True):
    b = Board()
    while not b.is_game_over():
        if b.turn:
            print(f"----- Move #{b.fullmove_number*2-1} -----")
            print("White to move:\n")
            print(b)
            print()
            legal_moves = list(b.generate_legal_moves())

            move = input("Your move (Enter 'h' to see all available legal moves):  ")
            if move == 'h':
                print("-"*27, "\n|     Available Moves     |", "\n-" + "-"*26)
                for m in legal_moves:
                    print(m)
                    continue
            try:
                move = chess.Move.from_uci(move)
            except ValueError as e:
                pass

            if move not in legal_moves:
                while move not in list(b.generate_legal_moves()):
                    move = input("Your move:  ")
                    try:
                        move = chess.Move.from_uci(move)
                    except ValueError as e:
                        print(f"ValueError: {e}... Try again.")
                        move = None

                    if move in legal_moves:
                        b.push(move)
                        break
            else:
                b.push(move)
            print()

        else:
            print(f"----- Move #{b.fullmove_number*2} -----")
            print("Black to move:\n")
            if VERBOSE_OUTPUT:
                print("_____ Candidate Moves _____")
            best_move, score = c.best_move(b, verbose=verbose)
            b.push(best_move)
            print()
            print(f"Best Move: {best_move}  with score = {round(score, 4)}")
            print()

    if b.is_checkmate():
        if b.result() == '1-0':
            print("You won!")
        else:
            print("You lost!")
    else:
        print("You tied!")

    return b

if __name__=='__main__':
    """
    c = ChessAgent()
    c.train_from_pgn('lichess_db_standard_rated_2014-07.pgn', n_games=1000000)
    c.save_model()
    """

    VERBOSE_OUTPUT = True
    c = load_model('models/mlp_regressor_default_lot_games.pkl')
