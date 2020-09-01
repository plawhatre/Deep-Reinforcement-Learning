import numpy as np
import os
import gym
from gym import wrappers
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import time 


class HiddenLayer:
	def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
		self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
		self.use_bias = use_bias
		if use_bias:
			self.b = tf.Variable(np.zeros(M2).astype(np.float32))
		self.f = f

	def forward(self, X):
		if self.use_bias:
			a = tf.matmul(X, self.W) + self.b
		else:
			a = tf.matmul(X, self.W)
		return self.f(a)

# APPROXIMATE: pi(a|s)
class PolicyModel:
	def __init__(self, dim, K, hidden_layer_sizes):
		# Create Graph of the Network
		self.layers = []
		M1 = dim
		for M2 in hidden_layer_sizes:
			layer = HiddenLayer(M1, M2)
			self.layers.append(layer)
			M1 = M2

		# Final Layer of the Network
		layer =HiddenLayer(M1, K, tf.nn.softmax, use_bias=False)

		self.layers.append(layer)

		#Inputs and Targets
		self.X = tf.placeholder(tf.float32, shape=(None, dim), name='X')
		self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
		self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

		# Calculate Output and Cost
		Z = self.X
		for layer in self.layers:
			Z = layer.forward(Z)
		pi_a_given_s = Z

		self.predict_op = pi_a_given_s
		selected_probs = tf.log(
			tf.reduce_sum(
				pi_a_given_s*tf.one_hot(self.actions,K),
				reduction_indices=[1]
				)
			)

		cost = -tf.reduce_sum(self.advantages*selected_probs)

		self.train_op = tf.train.AdagradOptimizer(10e-2).minimize(cost)

	def set_session(self, session):
		self.session = session

	def partial_fit(self, X, action, advantage):
		X = np.atleast_2d(X)
		action = np.atleast_1d(action)
		advantage = np.atleast_1d(advantage)

		self.session.run(
			self.train_op,
			feed_dict={
			self.X: X,
			self.actions: action,
			self.advantages: advantage,
			}
			)

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict={self.X: X})

	def sample_action(self, X):
		p = self.predict(X)[0]
		return np.random.choice(len(p), p=p)

# APPROXIMATE: V(S)
class ValueModel:
	def __init__(self, dim, hidden_layer_sizes):
		# Create Graph of the Network
		self.layers = []
		M1 = dim
		for M2 in hidden_layer_sizes:
			layer = HiddenLayer(M1, M2)
			self.layers.append(layer)
			M1 = M2

		# Final Layer of the Network
		layer =HiddenLayer(M1, 1, lambda x: x)
		self.layers.append(layer)

		#Inputs and Targets
		self.X = tf.placeholder(tf.float32, shape=(None, dim), name='X')
		self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

		# Calculate Output and Cost
		Z = self.X
		for layer in self.layers:
			Z = layer.forward(Z)
		Y_hat = tf.reshape(Z, [-1])
		self.predict_op = Y_hat

		cost = tf.reduce_sum(self.Y - Y_hat)

		self.train_op = tf.train.AdagradOptimizer(10e-5).minimize(cost)

	def set_session(self, session):
		self.session = session

	def partial_fit(self, X, Y):
		X = np.atleast_2d(X)
		Y = np.atleast_1d(Y)

		self.session.run(
			self.train_op,
			feed_dict={
			self.X: X,
			self.Y: Y,
			}
			)

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict={self.X: X})

# Temporal Difference 
def play_one_td(env, pmodel, vmodel, gamma):
	observation = env.reset()
	done = False
	totalreward = 0
	itr = 0
	while not done:
		action = pmodel.sample_action(observation)
		prev_observation = observation
		observation, reward, done, _ = env.step(action)

		totalreward += reward
		
		V_next = vmodel.predict(observation)
		G = reward + gamma*V_next
		advantage = G - vmodel.predict(prev_observation)
		pmodel.partial_fit(prev_observation, action, advantage)
		vmodel.partial_fit(prev_observation, G)

			
		itr += 1
	return totalreward

# Monte Carlo
def play_one_mc(env, pmodel, vmodel, gamma):
	observation = env.reset()
	done = False
	totalreward = 0
	itr = 0

	states = []
	actions = []
	rewards = []

	while not done:
		action = pmodel.sample_action(observation)
		prev_observation = observation
		observation, reward, done, _ = env.step(action)

		totalreward += reward
		
		V_next = vmodel.predict(observation)
		G = reward + gamma*V_next
		advantage = G - vmodel.predict(prev_observation)
		pmodel.partial_fit(prev_observation, action, advantage)
		vmodel.partial_fit(prev_observation, G)

		if done:
			reward = -200

		states.append(prev_observation)
		actions.append(action)
		rewards.append(reward)

		if reward == 1:
			totalreward += reward
			
		itr += 1

	returns = []
	advantages = []
	G = 0
	for s,r in zip(reversed(states),reversed(actions)):
		returns.append(G)
		advantages.append(G - vmodel.predict(s)[0])
		G = r + gamma*G
	returns.reverse()
	advantages.reverse()

	# Update the models
	pmodel.partial_fit(states, actions, advantages)
	vmodel.partial_fit(states, returns) 

	return totalreward

def plot_running_avg(totalrewards):
	N = len(totalrewards)
	print()
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(totalrewards[max(0,t-100):(t+1)])

	plt.plot(running_avg)
	plt.title('Running Average')
	plt.show()

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	dim = env.observation_space.shape[0]
	K = env.action_space.n
	pmodel = PolicyModel(dim, K, [])
	vmodel = ValueModel(dim, [10])
	init = tf.global_variables_initializer()
	session = tf.InteractiveSession()
	session.run(init)
	pmodel.set_session(session)
	vmodel.set_session(session)
	gamma = 0.99

	if not os.path.exists('Results'):
		os.mkdir('Results')
	os.chdir('Results')
	env = wrappers.Monitor(env, 'Policy_Gradient_Result', force=True)

	N = 500
	totalrewards = np.empty(N)
	for n in range(N):
		eps = 1/(0.1*n+1)
		totalreward = play_one_mc(env, pmodel, vmodel, gamma)
		# totalreward = play_one_td(env, pmodel, vmodel, gamma)
		totalrewards[n] = totalreward
		print(f'Episode : {n}, Total Reward: {totalreward}')

	plt.plot(totalrewards)
	plt.title('Total Reward')
	plt.show()
	# time.sleep(5)
	# plt.close()
	plot_running_avg(totalrewards)
	# time.sleep(5)
	# plt.close()
