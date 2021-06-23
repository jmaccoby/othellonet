from nodes import Pure
from boards import TTT

from copy import deepcopy
from time import time
from sys import stdin

import numpy as np

rng = np.random.default_rng()

class MCTS:

	def __init__(self, root, move_time):
		self.root = root
		self.next_move = root
		self.move_time = move_time

	def round(self, node):
		if node.board.finished:
			outcome = -node.board.winner
			node.update(outcome)
		elif not node.visits:
			outcome = node.evaluate()
			node.update(outcome)
		else:
			outcome = -self.round(node.next())
			node.update(outcome)
		#print(outcome)
		return outcome

	# def evaluate(self, board):
	# 	if not board.finished:
	# 		outcome = rollout(board.random)
	# 		return outcome
	# 	else:
	# 		return board.outcome

	def search(self):
		if self.root.board.finished:
			print('game is over')
			return None
		end_time = time() + self.move_time
		count = 0
		while  time() < end_time:
			self.round(self.root)
			count += 1
		self.next_move = self.root.select()
		# print('visit count: %d\nvalue: %d' % (self.next_move.visits, self.next_move.value))
		# possible = [move.value for move in self.root.searched]
		# print('possible values: %s' % ', '.join(str(i) for i in possible))
		# possible = [move.visits for move in self.root.searched]
		# print('possible visits: %s' % ', '.join(str(i) for i in possible))
		# possible = [move.ucb(self.root.visits) for move in self.root.searched]
		# print('possible ucb: %s' % ', '.join(str(i) for i in possible))
		# print(self.root.board.player)
		# self.next_move.board.display()
		#print(self.next_move.board.player)
		#print('search rounds: %d' % count)
		return self.next_move.board.last_move

	def progress_game(self, move):
		self.root = self.next_move.find_child(move)


class HumanPlayer:

	def __init__(self, board):
		self.board = board

	def search(self):
		print('player turn')
		for line in stdin:
			try:
				loc = [int(i) - 1 for i in line.split(', ')]
				if len(loc) < 2:
					print('invalid input')
				else:
					loc = tuple(loc[0:2])
					if self.board.validate_move(loc):
						self.board = self.board.execute_move(loc)
						return loc
					print('invalid input')

			except ValueError:
				print('invalid input')

	def progress_game(self, move):
		self.board = self.board.execute_move(move)

