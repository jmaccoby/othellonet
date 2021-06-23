from players import MCTS, HumanPlayer
from nodes import Pure
from boards import TTT
from sys import stdin

class Game:

	def __init__(self, player1, player2, board, first=0):
		self.player1 = player1
		self.player2 = player2
		self.board = board
		self.turn = first


	def step(self):
		if self.turn:
			row, col = self.player2.search()
			self.player1.progress_game((row, col))
		else:
			row, col = self.player1.search()
			self.player2.progress_game((row, col))
		self.board = self.board.execute_move((row, col))
		# self.board.display()
		# if (row, col) == (-1, -1):
		# 	print('player %d passes turn' % (self.turn + 1))
		# else:
		# 	print('player %d places on (%d, %d)' % (self.turn + 1, row + 1, col + 1))
		self.turn = not self.turn

	def play(self):
		print('player %i plays first' % (self.turn + 1))
		while not self.board.finished:
			self.step()
		print('game is over')
		return self.winner()


	def winner(self):
		if not self.board.winner:
			print('tie')
			return 0
		elif self.board.winner == 1:
			print('player %i wins' % (self.turn + 1))
			return self.turn + 1
		else:
			print('player %i wins' % ((not self.turn) + 1))
			return (not self.turn) + 1


