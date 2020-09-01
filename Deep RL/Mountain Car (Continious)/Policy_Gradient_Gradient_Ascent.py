import numpy as np
import os
import gym
from gym import wrappers	
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import time 
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

print(u'\u2716\u2716\u2716 Tunning Required\u2716\u2716\u2716')

class FeatureTransform:
	def __init__(self, env):
		observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
		scalar = StandardScaler()
		scalar.fit(observation_examples)

		featurizer = FeatureUnion([
			('rbf1', RBFSampler(gamma=0.05, n_components=500)),
			('rbf2', RBFSampler(gamma=1.0, n_components=500)),
			('rbf3', RBFSampler(gamma=0.5, n_components=500)),
			('rbf4', RBFSampler(gamma=0.1, n_components=500))
			])
		feature_example = featurizer.fit_transform(scalar.transform(observation_examples))

		self.dimension = feature_example.shape[1]
		self.scalar = scalar
		self.featurizer = featurizer

	def transform(self, observations):
		scaled = self.scalar.transform(observations)
		featurized = self.featurizer.transform(scaled)
		return featurized

class HiddenLayer:
	def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
		if zeros:
			self.W = tf.Variable(np.zeros((M1, M2), dtype=np.float32))
		else:
			self.W = tf.Variable(tf.random_normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32))

		self.use_bias = use_bias

		if use_bias:
			self.b = tf.Variable(np.zeros(M2).astype(np.float32))
		self.f = f

	def forward(self, X):
		if self.use_bias:
			activation = tf.matmul(X, self.W) + self.b
		else:
			activation = tf.matmul(X, self.W)
		return self.f(activation)

# APPROXIMATE: pi(a|s)
class PolicyModel:
	def __init__(self, dim, ft, hidden_layer_sizes=[]):
		# Create Graph of the Network
		self.ft = ft
		# self.dim = dim
		## Model of the Hidden Layer ##
		self.hidden_layers = []
		M1 = dim
		for M2 in hidden_layer_sizes:
			layer = HiddenLayer(M1, M2)
			self.hidden_layers.append(layer)
			M1 = M2

		# Final Layer of the Network (mean is unbounded hence identiy)
		self.mean_layer =HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
		
		# Final Layer of the Network (variance must be positive hence softplus)
		self.var_layer =HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)
		
		#Inputs and Targets
		self.X = tf.placeholder(tf.float32, shape=(None, dim), name='X')
		self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
		self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

		# get final hidden layer
		Z = self.X
		for layer in self.hidden_layers:
			Z = layer.forward(Z)

		mean = tf.reshape(self.mean_layer.forward(Z), [-1])
		var = tf.reshape(self.var_layer.forward(Z) + 1e-5, [-1])

		norm = tf.distributions.Normal(mean, var)

		self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)
		
		log_probs = norm.log_prob(self.actions)
		cost = -tf.reduce_sum(self.advantages*log_probs + 0.1*norm.entropy())
		self.train_op = tf.train.AdamOptimizer(10e-3).minimize(cost)

	def set_session(self, session):
		self.session = session

	def partial_fit(self, X, actions, advantages):
		X = np.atleast_2d(X)
		X = self.ft.transform(X)

		actions = np.atleast_1d(actions)
		advantages = np.atleast_1d(advantages)

		self.session.run(
			self.train_op,
			feed_dict={
			self.X: X,
			self.actions: actions,
			self.advantages: advantages
			})

	def predict(self, X):
		X = np.atleast_2d(X)
		X = self.ft.transform(X)
		return self.session.run(self.predict_op, feed_dict={self.X: X})

	def sample_action(self, X):
		p = self.predict(X)[0]
		return p

# APPROXIMATE: V(S)
class ValueModel:
	def __init__(self, dim, ft, hidden_layer_sizes=[]):
		self.ft = ft
		self.costs = []
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

		cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
		self.cost = cost
		self.train_op = tf.train.AdamOptimizer(10e-1).minimize(cost)

	def set_session(self, session):
		self.session = session

	def partial_fit(self, X, Y):
		X = np.atleast_2d(X)
		X = self.ft.transform(X)
		Y = np.atleast_1d(Y)

		self.session.run(
			self.train_op,
			feed_dict={
			self.X: X,
			self.Y: Y,
			}
			)
		cost = self.session.run(
			self.cost,
			feed_dict={
			self.X: X,
			self.Y: Y,
			}
			)
		self.costs.append(cost)

	def predict(self, X):
		X = np.atleast_2d(X)
		X = self.ft.transform(X)
		return self.session.run(self.predict_op, feed_dict={self.X: X})
	
def play_one(env, pmodel, vmodel, gamma):
	observation = env.reset()
	done = False
	totalreward = 0
	itr = 0
	while not done:
		action = pmodel.sample_action(observation)
		prev_observation = observation
		observation, reward, done, _ = env.step([action])
		totalreward += reward

		#update models
		V_next = vmodel.predict(observation)
		G = reward + gamma*V_next
		advantage = G - vmodel.predict(prev_observation)
		pmodel.partial_fit(prev_observation, action, advantage)
		vmodel.partial_fit(prev_observation, G)
		itr += 1
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
	env = gym.make('MountainCarContinuous-v0')
	ft = FeatureTransform(env)
	dim = ft.dimension
	pmodel = PolicyModel(dim, ft, [])
	vmodel = ValueModel(dim, ft, [])
	init = tf.global_variables_initializer()
	session = tf.InteractiveSession()
	session.run(init)
	pmodel.set_session(session)
	vmodel.set_session(session)
	gamma = 0.95

	# if not os.path.exists('Results'):
	# 	os.mkdir('Results')
	# os.chdir('Results')
	# env = wrappers.Monitor(env, 'Policy_Gradient_Gradient_Ascent_Result', force=True)

	N = 50
	totalrewards = np.empty(N)
	for n in range(N):
		totalreward = play_one(env, pmodel, vmodel, gamma)
		print(totalreward)
		totalrewards[n] = totalreward
		if n % 1 == 0:
		      print("episode:", n, "total reward: %.1f" % totalreward, "avg reward (last 100): %.1f" % totalrewards[max(0, n-100):(n+1)].mean())

	print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
	print(u'\u2716\u2716\u2716 Tunning Required\u2716\u2716\u2716')
	plt.plot(totalrewards)
	plt.title('Total Reward')
	plt.show()

	plot_running_avg(totalrewards)
