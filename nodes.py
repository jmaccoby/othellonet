from functools import cached_property
from math import log, inf

import numpy as np
import tensorflow as tf
from scipy.special import softmax

rng = np.random.default_rng()

class Pure:

	def __init__(self, board, exploration=0.5*2**0.5):
		self.visits = 0
		self.value = 0
		self.board = board
		self.c = exploration
		self.searched = []

	@cached_property
	def unsearched(self):
		return [Pure(b, self.c) for b in self.board.moves]

	def next(self):
		if self.unsearched:
			n = self.unsearched.pop(rng.integers(0, len(self.unsearched)))
			self.searched.append(n)
			return n
		best_value = -inf
		for child in self.searched:
			current_value = child.ucb(self.visits)
			# print('Current value: %0.3f' % current_value)
			# print('Best value: %0.3f' % best_value)
			#child.board.display()
			if current_value > best_value:
				best_value = current_value
				best = [child]
			elif current_value == best_value:
				best.append(child)
		#print('Best value: %0.3f' % best_value)
		return best[rng.integers(0, len(best))]

	def update(self, win):
		self.visits += 1
		if win == 1:
			self.value += 1
		#print('updating current value %d with %d' % (self.value, win))
		#self.value += win

	def select(self):
		most_visits = 0
		for child in self.searched:
			current = child.visits
			# print('Visits: %0.3f' % current)
			# print('Value: %0.3f' % child.value)
			# print('Total value: %d' % self.value)
			# print(child.board.last_move)
			if current > most_visits:
				most_visits = current
				best = [child]
			elif current == most_visits:
				best.append(child)
		# print('Best: %0.3f' % most_visits)
		# print('Most wins: % 0.3f' % best.value)
		best = best[rng.integers(0, len(best))]
		#best.board.display()
		return best

	def find_child(self, move):
		for child in self.searched:
			if child.board.last_move == move:
				return child
		for child in self.unsearched:
			if child.board.last_move == move:
				return child
		print('call the police!')

	def ucb(self, total_visits):
		return self.value/self.visits + self.c * (log(total_visits)/self.visits) ** 0.5

	#1 represents win, -1 for loss, 0 for tie
	def rollout(self, b):
		#b.display()
		if not b.finished:
			#negate return value, as every other move is from opponent's perspective
			return  -self.rollout(b.random())
		#tie
		# b.display()
		# print(b.winner)
		return b.winner

	def evaluate(self):
		outcome = -self.rollout(self.board)
		#print(outcome)
		return outcome


class Net:

	def __init__(self, board, net, generate_data=False, train_data=None, move_count=0, exploration=1):
		self.board = board
		self.net = net
		self.visits = 0
		self.value = 0
		self.c = exploration
		self.generate_data = generate_data
		self.train_data = train_data
		self.move_count = move_count


	@cached_property
	def children(self):
		return [Net(b, self.net, self.generate_data, self.train_data, self.move_count-1, self.c) for b in self.board.moves]

	def next(self):
		policy = [self.ucb(child, prob) for (child, prob) in zip(self.children, self.prior_probs)]
		return self.children[np.argmax(policy)]

	def update(self, value):
		self.visits += 1
		self.value += value

	def select(self):
		policy = [1/len(self.children) if self.visits < 2 else child.visits/(self.visits - 1) for child in self.children]
		if self.generate_data:
			self.train_data.append((self.board, policy))
		#select best move
		if self.move_count < 1:
			return self.children[np.argmax(policy)]
		#sample from policy for the first few turns during self-play
		return rng.choice(self.children, p=policy)


	def find_child(self, move):
		for child in self.children:
			if child.board.last_move == move:
				return child
		print('call the police!')

	def evaluate(self):
		value, probs = self.net(self.board.bin_arrays, training=False)
		probs = probs[0]
		valid_probs = []
		pass_prob = probs[64]
		probs = np.reshape(probs[:64], (8, 8))
		for child in self.children:
			i, j = child.board.last_move
			if (i, j) == (-1, -1):
				valid_probs = [pass_prob]
			else:
				valid_probs.append(probs[i][j])
		self.prior_probs = [prob/sum(valid_probs) for prob in valid_probs]
		return -value

	# modified PUCT
	def ucb(self, child, prob):
		q = 0 if not child.visits else child.value/child.visits
		return q + self.c * prob * ((self.visits - 1) ** 0.5 )/ (1 + child.visits)



