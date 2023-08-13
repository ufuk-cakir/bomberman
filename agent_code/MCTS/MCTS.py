
EPS = 1e-8


#https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
class MCTS():
    def __init__(self,game,nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.q_values_s_a = {}  # stores Q values for s,a (as defined in the paper)
        self.times_visited_edge_a_s = {}  # stores #times edge s,a was visited
        self.times_visited_board_s = {}  # stores #times board s was visited
        self.initial_policy = {}  # stores initial policy (returned by neural net)

        self.game_ended_s = {}  # stores game.getGameEnded ended for board s
        self.valid_moves_s = {}  # stores game.getValidMoves for board s 

    def get_action_prob(self, canonical_board, temp =1):
