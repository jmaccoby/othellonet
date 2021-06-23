from players import MCTS, HumanPlayer
from nodes import Pure, Net
from boards import TTT
from game import Game
from boards import Othello, othello_start

from tensorflow import keras

from sys import stdin

def test_ttt(time1=5, time2 = 5):
	player1 = MCTS(Pure(TTT()), time1)
	player2 = MCTS(Pure(TTT(player=1)), time2)
	return Game(player1, player2, TTT())

def test_othello(time1=30, time2=30):
	player1 = MCTS(Pure(othello_start()), time1)
	player2 = MCTS(Pure(othello_start()), time2)
	return Game(player1, player2, othello_start())

def test_human_othello(time=30, player=0):
	if player:
		player1 = HumanPlayer(othello_start())
		player2 = MCTS(Pure(othello_start()), time)
	else:
		player1 = MCTS(Pure(othello_start()), time)
		player2 = HumanPlayer(othello_start())
	return Game(player1, player2, othello_start())

def test_othello_net(time=10, player=0):
	net = keras.models.load_model('networks/othello_best.h5')
	board = othello_start()
	player1 = MCTS(Net(board, net), time)
	player2 = HumanPlayer(othello_start())
	return Game(player1, player2, othello_start(), player)

def test_human_ttt(time=30, player=0):
	if player:
		player1 = HumanPlayer()
		player2 = MCTS(Pure(TTT()), time)
	else:
		player1 = MCTS(Pure(TTT()), time)
		player2 = HumanPlayer()
	return Game(player1, player2, TTT())

root = Pure(TTT())

t = test_ttt()

o = test_othello(5, 5)

h = test_human_othello()

oth = othello_start()

m = MCTS(Pure(TTT()), 5)

def io_test():
	for line in stdin:
		loc = [int(i) for i in line.split(', ')]
		if len(loc) < 2:
			print('invalid input')
		else:
			print('selected move: (%d, %d)' % tuple(loc[0:2]))
