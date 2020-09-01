import numpy as np
import matplotlib.pyplot as plt
import os
import gym
from gym import wrappers
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class HiddenLayer:
	#we have to keep track of params since we have to copy network ahead
	def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
		self.W = tf.Variable(tf.random_normal(shape=(M1,M2)))
		self.params = [self.W]
		self.use_bias = use_bias
		if use_bias:
			self.b = tf.Variable(np.zeros(M2).astype(np.float32))
			self.params.append(self.b)
		self.f = f

	def forward(self, X):
		if self.use_bias:
			return self.f(tf.matmul(X, self.W) + self.b)
		else:
			return self.f(tf.matmul(X, self.W))

class DQN:
	def __init__(self, dim, K, hidden_layers_sizes, gamma, max_exp=10000, min_exp=100, batch_size=32):
		self.K = K

		#hidden layers
		self.layers = []
		M1 = dim
		for M2 in hidden_layers_sizes:
			layer = HiddenLayer(M1, M2)
			self.layers.append(layer)
			M1 = M2
		
		# Final layer
		layer = HiddenLayer(M1, K, lambda x: x)
		self.layers.append(layer)

		#collect params for copy
		self.params = []
		for layer in self.layers:
			self.params += layer.params

		#inputs and targets
		self.X  = tf.placeholder(tf.float32, shape=(None, dim), name="X")
		self.G = tf.placeholder(tf.float32, shape=(None,), name="G") 
		self.actions = tf.placeholder(tf.int32, shape=(None,), name="actions") 

		#output and cost
		Z = self.X
		for layer in self.layers:
			Z = layer.forward(Z)
		Y_hat =Z

		self.predict_op = Y_hat

		selected_action_values = tf.reduce_sum(
			Y_hat * tf.one_hot(self.actions, K),
			reduction_indices=[1]
			)

		cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
		self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

		#create replay memory
		self.exp =  {'s':[], 'a':[], 'r':[], 's2':[], 'done':[]}
		self.max_exp = max_exp
		self.min_exp = min_exp
		self.batch_size = batch_size
		self.gamma = gamma

	def set_session(self, session):
		self.session = session

	def copy_from(self, other):
		# collect all ops
		ops = []
		my_params = self.params
		other_params = other.params

		for p,q in zip(my_params, other_params):
			actual = self.session.run(q)
			op = p.assign(actual)
			ops.append(op)

		self.session.run(ops)

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict={
			self.X: X
			})

	def train(self, target_network):
		if len(self.exp['s']) < self.min_exp:
			#don't do anything
			return
		idx = np.random.choice(len(self.exp['s']), size=self.batch_size, replace=False)
		states = [self.exp['s'][i] for i in idx] 
		actions = [self.exp['a'][i] for i in idx] 
		rewards = [self.exp['r'][i] for i in idx] 
		next_states = [self.exp['s2'][i] for i in idx]
		dones = [self.exp['done'][i] for i in idx]
		next_Q = np.max(target_network.predict(next_states), axis=1)
		targets =  [r + self.gamma*next_q if not done else r for r,next_q,done in zip(rewards,next_Q, dones)]

		# call optimizers
		self.session.run(self.train_op,
			feed_dict={
			self.X: states,
			self.G: targets,
			self.actions: actions 
			})
	def add_experience(self, s, a, r, s2, done):
		if len(self.exp['s']) >= self.max_exp:
			self.exp['s'].pop(0)
			self.exp['a'].pop(0)
			self.exp['r'].pop(0)
			self.exp['s2'].pop(0)
			self.exp['done'].pop(0)
		self.exp['s'].append(s)
		self.exp['a'].append(a)
		self.exp['r'].append(r)
		self.exp['s2'].append(s2)
		self.exp['done'].append(done)

	def sample_action(self, x, eps):
		if np.random.random()< eps:
			return np.random.choice(self.K)
		else:
			X = np.atleast_2d(x)
			return np.argmax(self.predict(X)[0])
def play_one(env, model, tmodel, eps, gamma, copy_period):
	observation = env.reset()
	done = False
	totalreward = 0
	itr = 0
	while not done:
		action = model.sample_action(observation, eps)
		prev_observation = observation
		observation, reward, done, info = env.step(action)

		totalreward += reward

		model.add_experience(prev_observation, action, reward, observation, done)

		model.train(tmodel)
		itr += 1

		if itr % copy_period == 0:
			tmodel.copy_from(model)

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

	#Environment
	env = gym.make('CartPole-v0')
	gamma = 0.99
	copy_period = 50
	
	#Network
	dim = len(env.observation_space.sample())
	K = env.action_space.n
	sizes = [200,200]
	model = DQN(dim, K, sizes, gamma)
	tmodel = DQN(dim, K, sizes, gamma)
	
	#Session
	init = tf.global_variables_initializer()
	session = tf.InteractiveSession()
	session.run(init)
	model.set_session(session)
	tmodel.set_session(session)
	
	#Directory to save results
	if not os.path.exists('Results'):
		os.mkdir('Results')
	os.chdir('Results')
	env = wrappers.Monitor(env, 'Policy_Gradient_Hill_Climbing_Result', force=True)

	N = 500
	totalrewards = np.empty(N)
	costs = np.empty(N)
	for n in range(N):
		eps = 1.0/np.sqrt(n+1)
		totalreward = play_one(env, model, tmodel, eps, gamma, copy_period)
		totalrewards[n] = totalreward
		if n % 100 == 0:
			print("episode:", n,
				"total reward:", totalreward,
				"eps:", eps,
				"avg reward (last 100):",totalrewards[max(0, n-100):(n+1)].mean())

	print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
	print("total steps:", totalrewards.sum())

	plt.plot(totalrewards)
	plt.title("Rewards")
	plt.show()

	plot_running_avg(totalrewards)