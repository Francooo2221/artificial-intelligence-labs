from connect4 import Connect4
import copy


class AlphaBetaAgent:
    def __init__(self, token: str, depth: int = 5, heuristics=True):
        self.my_token = token
        self.depth = depth
        self.heuristics = heuristics

    def decide(self, connect4: Connect4, ):
        alfa = -float('inf')
        max_value = -float('inf')
        move = None
        for n_column in connect4.possible_drops():
            new_connect4 = copy.deepcopy(connect4)
            new_connect4.drop_token(n_column)
            value = self.alphabeta(new_connect4, self.depth - 1, False, alfa)
            if value > max_value:
                max_value = value
                move = n_column
        return move

    def alphabeta(self, connect4: Connect4, depth: int, maximizing: bool, alpha=-float('inf'), beta=float('inf')):
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1
            elif connect4.wins is None:
                return 0
            else:
                return -1

        if depth == 0:
            if self.heuristics==True:
                return self.heuristic(connect4)
            else:
                return 0

        if maximizing:
            value = -float('inf')
            for n_column in connect4.possible_drops():
                new_connect4 = copy.deepcopy(connect4)
                new_connect4.drop_token(n_column)
                value = max(value, self.alphabeta(new_connect4, depth - 1, not maximizing, alpha, beta))
                alpha = max(alpha, value)
                if value >= beta:
                    break
            return value
        else:
            value = float('inf')
            for n_column in connect4.possible_drops():
                new_connect4 = copy.deepcopy(connect4)
                new_connect4.drop_token(n_column)
                value = min(value, self.alphabeta(new_connect4, depth - 1, not maximizing, alpha, beta))
                beta = min(beta, value)
                if value <= alpha:
                    break
            return value

    def heuristic(self, connect4: Connect4):
        weights = [0, 5, 10, 15]
        points = 0
        total = 0
        for four in connect4.iter_fours():
            total += 1
            our = four.count(self.my_token)
            empty = four.count('_')
            rival = 4 - our - empty
            if rival != 0:
                continue
            points += weights[our]
        return points / (total * weights[-1])



