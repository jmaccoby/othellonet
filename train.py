import numpy as np
import tensorflow as tf
from tensorflow import keras
from game import Game
from nodes import Net
from boards import Othello, othello_start
from players import MCTS
import pickle
import numpy as np

rng = np.random.default_rng()


def self_play(num_games=50, move_time=1):
	player1 = keras.models.load_model('networks/othello_best.h5')
	player2 = keras.models.load_model('networks/othello_best.h5')
	board = othello_start()
	with open('train_data.txt', 'rb') as fp:
		examples = pickle.load(fp)
	for i in range(num_games):
		print('\n\n\nstarting game %d' % (i + 1))
		train_data = []
		search1 = MCTS(Net(board, player1, True, train_data, 20), move_time)
		search2 = MCTS(Net(board, player2, True, train_data, 20), move_time)
		game = Game(search1, search2, board)
		game.play()
		outcome = float(game.board.winner)
		examples.extend([(board.bin_arrays, board.fill_policy(policy), outcome if board.player == game.turn else -outcome) 
			for (board, policy) in train_data])
	with open('train_data.txt', 'wb') as fp:
		pickle.dump(examples, fp)

def prep_data():
	with open('train_data.txt', 'rb') as fp:
		data = pickle.load(fp)
	boards, policy, outcome = zip(*data)
	X = np.concatenate(boards)
	value = np.asarray(outcome)
	pol = np.asarray(policy)
	return X, value, pol

def evaluate_net(num_games=40, move_time=1, win_threshold=24):
	base_net = keras.models.load_model('networks/othello_best.h5')
	new_net = keras.models.load_model('networks/othello_current.h5')
	board = othello_start()
	wins = 0
	losses = 0
	ties = 0
	for i in range(num_games):
		print('starting game %d' % (i + 1))
		print('updated network score:')
		print('wins: %d\tlosses: %d\tties: %d' % (wins, losses, ties))
		base_search = MCTS(Net(board, base_net, move_count=5), move_time)
		new_search = MCTS(Net(board, new_net, move_count=5), move_time)
		game = Game(base_search, new_search, board, rng.integers(0, 2))
		winner = game.play()
		if winner == 2:
			print('updated net wins')
			wins += 1
		elif winner == 1:
			print('updated net loses')
			losses += 1
		else:
			print('tie')
			ties += 1
		print('\n\n\n')
		if wins >= win_threshold:
			break
		if losses > num_games - win_threshold:
			print('updated net is no longer able to win enough games')
			print('ending evaluation early')
			break
	print('final score:')
	print('wins: %d\tlosses: %d\tties: %d' % (wins, losses, ties))
	if wins >= win_threshold:
		print('updated network accepted as new baseline')
		print('saving updated network')
		new_net.save('networks/othello_best.h5')

def update_net():
	board, val, pol = prep_data()
	net = keras.models.load_model('networks/othello_best.h5')
	history = net.fit(board, [val, pol], epochs=10, batch_size=64)
	net.save('networks/othello_current.h5')

def training_iter(iterations, self_play_rounds=100, eval_rounds=40, win_threshold=24, move_time=1):
	for i in range(iterations):
		self_play(self_play_rounds, move_time)
		update_net()
		evaluate_net(eval_rounds, move_time, win_threshold)
	net = keras.models.load_model('networks/othello_best.h5')
	print('creating network checkpoint')
	net.save('networks/othello_checkpoint.h5')

