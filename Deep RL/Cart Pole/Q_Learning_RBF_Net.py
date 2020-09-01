import numpy as np
import os, gym
from gym import wrappers
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class SGDRegressor:
	def __init__(self, dim):
		self.w = np.random.randn(dim) / np.sqrt(dim)
		self.lr = 10e-2

	def partial_fit(self, X, Y):
		self.w += self.lr*(Y - X.dot(self.w)).dot(X)

	def predict(self, X):
		return X.dot(self.w)

class FeatureTransform:
	def __init__(self, env):
		'''
		we are not sampling from observation space 
		rather from a uniform distribution as its 
		state space extend to infinity.
		'''
		observation_examples = np.random.random((20000,4))*2-2
		scalar = StandardScaler()
		scalar.fit(observation_examples)

		featurizer = FeatureUnion([
			('rbf1', RBFSampler(gamma=0.05, n_components=1000)),
			('rbf2', RBFSampler(gamma=1.0, n_components=1000)),
			('rbf3', RBFSampler(gamma=0.5, n_components=1000)),
			('rbf4', RBFSampler(gamma=0.1, n_components=1000))
			])
		feature_example = featurizer.fit_transform(scalar.transform(observation_examples))

		self.dimension = feature_example.shape[1]
		self.scalar = scalar
		self.featurizer = featurizer

	def transform(self, observations):
		scaled = self.scalar.transform(observations)
		featurized = self.featurizer.transform(scaled)
		return featurized

class Model:
	def __init__(self, env, feature_transformer, learning_rate):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer
		for i in range(env.action_space.n):
			# custom sgd class has dimension instead of learning rate during instantiation
			model = SGDRegressor(feature_transformer.dimension)	
			self.models.append(model)

	def predict(self, s):
		X = self.feature_transformer.transform(np.atleast_2d(s))
		pred = np.array([m.predict(X) for m in self.models])
		return pred 

	def update(self, s, a, G):
		X = self.feature_transformer.transform(np.atleast_2d(s))
		self.models[a].partial_fit(X, [G])

	def sample_action(self, p, eps):
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(p))

def play_one(model, env, eps, gamma):
	observation = env.reset()
	done = False
	total_reward = 0
	itr = 0
	while not done:
		action = model.sample_action(observation, eps)
		prev_observation =observation
		observation, reward, done, _ = env.step(action)

		G = reward + gamma*np.max(model.predict(observation))
		model.update(prev_observation,action,G)
		total_reward += reward
		itr += 1

	return total_reward

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
	ft = FeatureTransform(env)
	model = Model(env, ft, "constant")
	gamma = 0.99

	if not os.path.exists('Results'):
		os.mkdir('Results')
	os.chdir('Results')
	env = wrappers.Monitor(env, 'Q_Learning_RBF_Net_Result', force=True)

	N = 500
	totalrewards = np.empty(N)
	for n in range(N):
		eps = 1/(0.1*n+1)
		totalreward = play_one(model, env, eps, gamma)
		totalrewards[n] = totalreward
		print(f'Episode : {n}, Total Reward: {totalreward}')

	plt.plot(totalrewards)
	plt.title('Total Reward')
	plt.show()

	plot_running_avg(totalrewards)
