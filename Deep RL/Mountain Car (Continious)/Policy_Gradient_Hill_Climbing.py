import numpy as np
import os
import gym
from gym import wrappers
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import time 
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class FeatureTransform:
	def __init__(self, env):
		'''
		we are not sampling from observation space 
		rather from a uniform distribution as its 
		state space extend to infinity.
		'''
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
			self.W = tf.Variable(np.zeros((M1, M2)).astype(np.float32))
		else:
			self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))

		self.params = [self.W]
		self.use_bias = use_bias

		if use_bias:
			self.b = tf.Variable(np.zeros(M2).astype(np.float32))
			self.params.append(self.b)
		self.f = f

	def forward(self, X):
		if self.use_bias:
			activation = tf.matmul(X, self.W) + self.b
		else:
			activation = tf.matmul(X, self.W)
		return self.f(activation)

# APPROXIMATE: pi(a|s)
class PolicyModel:
	def __init__(self, ft, dim, hidden_layer_sizes_mean=[], hidden_layer_sizes_var=[]):
		# Create Graph of the Network
		self.ft = ft
		self.dim = dim
		self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
		self.hidden_layer_sizes_var = hidden_layer_sizes_var

		## Model of the Mean ##
		self.mean_layers = []
		M1 = dim
		for M2 in hidden_layer_sizes_mean:
			layer = HiddenLayer(M1, M2)
			self.mean_layers.append(layer)
			M1 = M2

		# Final Layer of the Network (mean is unbounded hence identiy)
		layer =HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
		self.mean_layers.append(layer)

		## Model of the Variance ##
		self.var_layers = []
		M1 = dim
		for M2 in hidden_layer_sizes_var:
			layer = HiddenLayer(M1, M2)
			self.var_layers.append(layer)
			M1 = M2

		# Final Layer of the Network (variance must be positive hence softplus)
		layer =HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)
		self.var_layers.append(layer)

		#params
		self.params = []
		for layer in (self.mean_layers + self.var_layers):
			self.params += layer.params

		#Inputs and Targets
		self.X = tf.placeholder(tf.float32, shape=(None, dim), name='X')
		self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
		self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

		# Calculate Output
		def get_output(layers):
			Z = self.X
			for layer in layers:
				Z = layer.forward(Z)
			return tf.reshape(Z, [-1])

		mean = get_output(self.mean_layers)
		var = get_output(self.var_layers)

		norm = tf.distributions.Normal(mean, var)

		self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)
		
	def set_session(self, session):
		self.session = session

	def init_vars(self):
		init_op = tf.variables_initializer(self.params)
		self.session.run(init_op)

	def predict(self, X):
		X = np.atleast_2d(X)
		X = self.ft.transform(X)
		return self.session.run(self.predict_op, feed_dict={self.X: X})

	def sample_action(self, X):
		p = self.predict(X)[0]
		return p

	# Thse functiona are required for hill climbing procedure
	def copy(self):
		clone = PolicyModel(self.ft, self.dim, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var)
		clone.set_session(self.session)
		clone.init_vars()
		clone.copy_from(self)
		return clone

	def copy_from(self, other):
		ops = []
		my_params = self.params
		other_params = other.params
		for p, q in zip(my_params, other_params):
			# get the value
			actual = self.session.run(q)
			# set the value
			op = p.assign(actual)
			ops.append(op)
		self.session.run(ops)

	def perturb_params(self):
		ops = []
		for p in self.params:
			v = self.session.run(p)
			noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0])*5
			if np.random.random() < 0.1:
				op = p.assign(noise)
			else:
				op = p.assign(v + noise)
			ops.append(op)
		self.session.run(ops)
	
def play_one(env, pmodel):
	observation = env.reset()
	done = False
	totalreward = 0
	itr = 0
	while not done:
		action = pmodel.sample_action(observation)
		observation, reward, done, _ = env.step([action])
		totalreward += reward
		itr += 1
	return totalreward

def play_multiple_episode(env, itr, pmodel, gamma, print_itrs=True):
	totalrewards = np.empty(itr)
	for i in range(itr):
		totalrewards[i] = play_one(env, pmodel)
		if print_itrs:
			print(i, "Average as of now:", totalrewards[:(i+1)].mean())

	avg_totalrewards = totalrewards.mean()
	print("avg totalrewards:", avg_totalrewards)
	return avg_totalrewards

def random_search(env, pmodel, gamma):
	print('__________________________ Random Search Starts__________________________')
	total_rewards = []
	best_avg_totalrewards = float('-inf')
	best_pmodel = pmodel
	num_episode_per_param_test = 3
	for t in range(100):
		tmp_model = best_pmodel.copy()
		tmp_model.perturb_params()
		avg_totalrewards = play_multiple_episode(env, num_episode_per_param_test, tmp_model, gamma)
		total_rewards.append(avg_totalrewards)
		print(f'Iteration Random Search: {t}, Avg. Total Reward: {avg_totalrewards}')
		if avg_totalrewards > best_avg_totalrewards:
			best_avg_totalrewards = avg_totalrewards
			best_pmodel = tmp_model

	print('__________________________ Random Search Ends__________________________')
	return total_rewards, best_pmodel

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

	pmodel = PolicyModel(ft, dim, [], [])
	session = tf.InteractiveSession()
	pmodel.set_session(session)
	pmodel.init_vars()
	gamma = 0.99

	if not os.path.exists('Results'):
		os.mkdir('Results')
	os.chdir('Results')
	env = wrappers.Monitor(env, 'Policy_Gradient_Hill_Climbing_Result', force=True)

	total_rewards, pmodel = random_search(env, pmodel, gamma)
	print('Maximum total rewards(Random Search)', np.max(total_rewards))
	avg_totalrewards = play_multiple_episode(env, 100, pmodel, gamma)
	print('Best Model:Avergae total rewards', avg_totalrewards)

	plt.plot(total_rewards)
	plt.title('Total Reward')
	plt.show()

	plot_running_avg(total_rewards)
