from functools import cached_property

import numpy as np
import tensorflow as tf

rng = np.random.default_rng()

#Tic-Tac-Toe board
#0 for empty space, 1 for player 1, 2 for player 2
class TTT:

	def __init__(self, board=np.zeros((3, 3), np.int8), player=0, last_move=None):
		self.board = board
		self.player = player
		self.last_move = last_move

	@cached_property
	def moves(self):
		if self.finished:
			return []
		rows, cols = self.board.shape
		m = []
		for i in range(rows):
			for j in range(cols):
				if not self.board[i, j]:
					m.append(self.next_state(i, j))
		return m

	def next_state(self, row, col):
		new_board = np.copy(self.board)
		new_board[row, col] = self.player + 1
		return TTT(new_board, not self.player, (row, col))

	def random(self):
		if self.finished:
			return None
		return self.moves[rng.integers(0, len(self.moves))]

	def execute_move(self, loc):
		for move in self.moves:
			if move.last_move == loc:
				return move
		print('missing board state')

	#finished represents a finished game
	#winner is from the perspective of the CURRENT player
	#1 for win, -1 for loss, 0 for tie
	@cached_property
	def finished(self):
	#check for current player's win
		b = self.board - self.player - 1
		for i in range(3):
			if not b[i, 0] and not b[i, 1] and not b[i, 2]:
				self.winner = 1
				return 1
			if not b[0, i] and not b[1, i] and not b[2, i]:
				self.winner = 1
				return 1
		if not b[0, 0] and not b[1, 1] and not b[2, 2]:
			self.winner = 1
			return 1
		if not b[0, 2] and not b[1, 1] and not b[2, 0]:
			self.winner = 1
			return 1

		#check for opponent's win
		b = self.board - (not self.player) - 1
		for i in range(3):
			if not b[i, 0] and not b[i, 1] and not b[i, 2]:
				self.winner = -1
				return 1
			if not b[0, i] and not b[1, i] and not b[2, i]:
				self.winner = -1
				return 1
		if not b[0, 0] and not b[1, 1] and not b[2, 2]:
			self.winner = -1
			return 1
		if not b[0, 2] and not b[1, 1] and not b[2, 0]:
			self.winner = -1
			return 1
		#check for tie
		if np.all(self.board):
			self.winner = 0
			return 1
		return 0

	def display(self):
		print(self.board)
		print('current player: %d' % (self.player + 1))




#Othello board
#player 1 is black, player 2 is white
#0 is empty space, 1 for black piece, 2 for white piece
class Othello:

	def __init__(self, board, player=0, last_move=(-1, -1)):
		self.player = player
		self.board = board
		self.last_move = last_move

	@cached_property
	def bin_arrays(self):
		array1 = [[1 if self.board[i][j] == (self.player + 1) else 0 for j in range(8)] for i in range(8)]
		array2 = [[1 if self.board[i][j] == ((not self.player) + 1) else 0 for j in range(8)] for i in range(8)]
		#return tf.expand_dims(tf.transpose(tf.convert_to_tensor((array1, array2)), perm=[1, 2, 0]), axis=0)
		return np.expand_dims(np.transpose(np.asarray((array1, array2)), (1, 2, 0)), 0)

	@cached_property
	def moves(self):
		m = self.check_moves()
		if not m and self.opp_moves:
			return [Othello(np.copy(self.board), not self.player)]
		return m

	@cached_property
	def opp_moves(self):
		return self.check_moves(opponent=True)

	def fill_policy(self, partial_policy):
		full_policy = [0] * 65
		for move, prob in zip(self.moves, partial_policy):
			i, j = move.last_move
			if (i, j) == (-1, -1):
				full_policy[64] = 1
			else:
				full_policy[i * 8 + j] = prob
		return full_policy

	def check_moves(self, opponent=False):
		perspective = self.player
		if opponent:
			perspective = not self.player
		rows, cols = self.board.shape
		potential_moves = []
		for i in range(rows):
			for j in range(cols):
				if not self.board[i, j]:
					adj = self.check_adjacent(i, j, perspective)
					if adj:
						potential_moves.append((i, j, adj))
		#print(potential_moves)
		legal_moves = self.next_state(potential_moves, perspective)
		return legal_moves

	def check_flank(self, row, col, i, j, new_board, perspective):
		row += i
		col +=j
		if row < 0 or row > 7 or col < 0 or col > 7:
			return False, new_board
		if not self.board[row, col]:
			return False, new_board
		if self.board[row, col] == perspective + 1:
			return True, new_board
		if self.board[row, col] == (not perspective) + 1:
			valid, new_board = self.check_flank(row, col, i, j, new_board, perspective)
			if valid:
				new_board[row, col] = perspective + 1
			return valid, new_board
		print('i missed something')

	def next_state(self, potential_moves, perspective):
		legal_moves = []
		for row, col, directions in potential_moves:
			valid = False
			new_board = np.copy(self.board)
			for i, j in directions:
				update, new_board = self.check_flank(row, col, i, j, new_board, perspective)
				if update:
					valid = True
			if valid:
				new_board[row, col] = perspective + 1
				legal_moves.append(Othello(new_board, not perspective, (row, col)))
		return legal_moves

	def check_adjacent(self, row, col, perspective):
		adj = []
		check = [(i - 1, j - 1) for i in range(3) for j in range(3)]
		for i, j in check:
			iloc, jloc = i + row, j + col
			if iloc >= 0 and iloc < 8 and jloc >= 0 and jloc < 8:
				if self.board[iloc, jloc] == (not perspective) + 1:
					adj.append((i, j))
		return adj

	def score(self):
		player = 0
		opponent = 0
		for row in self.board:
			for piece in row:
				if piece == self.player + 1:
					player += 1
				elif piece == (not self.player) + 1:
					opponent += 1
		return 1 if player > opponent else -1 if opponent > player else 0

	def execute_move(self, loc):
		for move in self.moves:
			if move.last_move == loc:
				return move
		print('missing board state')

	def validate_move(self, loc):
		for move in self.moves:
			if move.last_move == loc:
				return True
		return False

	@cached_property
	def finished(self):
		if self.moves or self.opp_moves:
			return False
		self.winner = self.score()
		return True

	def random(self):
		if self.finished:
			return None
		return self.moves[rng.integers(0, len(self.moves))]

	def display(self):
		print(self.board)
		#print(self.player)


def othello_start():
	b = np.zeros((8, 8), np.int8)
	b[3, 3] = 2
	b[3, 4] = 1
	b[4, 3] = 1
	b[4, 4] = 2
	return Othello(b)

