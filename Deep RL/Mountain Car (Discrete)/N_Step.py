import numpy as np
import os
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class SGDRegressor:
	def __init__(self, **kwargs):
		self.w = None
		self.lr = 10e-3

	def partial_fit(self, X, Y):
		if np.array(self.w == None).any():
			dim = np.shape(X)[1]
			self.w = np.random.randn(dim) / np.sqrt(dim)
		self.w += self.lr*(Y - X.dot(self.w)).dot(X)

	def predict(self, X):
		return X.dot(self.w)

class FeatureTransform:
	def __init__(self, env):
		observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
		scalar = StandardScaler()
		scalar.fit(observation_examples)

		featurizer = FeatureUnion([
			('rbf1', RBFSampler(gamma=5.0, n_components=500)),
			('rbf2', RBFSampler(gamma=2.1, n_components=500)),
			('rbf3', RBFSampler(gamma=1.0, n_components=500)),
			('rbf4', RBFSampler(gamma=0.5, n_components=500))
			])
		featurizer.fit(scalar.transform(observation_examples))

		self.scalar = scalar
		self.featurizer = featurizer

	def transform(self, observations):
		scaled = self.scalar.transform(observations)
		featurized = self.featurizer.transform(scaled)
		return featurized

class Model:
	def __init__(self, env, feature_transformer):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer
		for i in range(env.action_space.n):
			model = SGDRegressor()
			model.partial_fit(feature_transformer.transform([env.reset()]) , [0])
			self.models.append(model)

	def predict(self, s):
		X = self.feature_transformer.transform([s])
		pred = np.array([m.predict(X) for m in self.models])
		return pred 

	def update(self, s, a, G):
		X = self.feature_transformer.transform([s])
		self.models[a].partial_fit(X, [G])

	def sample_action(self, p, eps):
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(p))
def play_one(model, eps, gamma, n=5):
	observation = model.env.reset()
	done = False
	totalreward = 0
	states = []
	actions = []
	rewards = []
	itr = 0
	mutiplier = np.array([gamma])**np.arange(n)

	while not done:
		action = model.sample_action(observation, eps)
		states.append(observation)
		actions.append(action)
		prev_observation = observation
		observation, reward, done, _ = model.env.step(action)
		rewards.append(reward)

		if len(rewards) >= n:
			return_sum = mutiplier.dot(rewards[-n:])
			G = return_sum + (gamma**n)*np.max(model.predict(observation)[0])
			model.update(states[-n], actions[-n], G)

		totalreward += reward
		itr += 1 

	states = states[-n+1:]
	actions = actions[-n+1:]
	rewards = rewards[-n+1:]

	#According to documentation the task is learned when the position reaches 0.5
	if observation[0] >= 0.5:
		while len(rewards) > 0:
			G = mutiplier[:len(rewards)].dot(rewards)
			model.update(states[0], actions[0], G)
			states.pop(0)
			actions.pop(0)
			rewards.pop(0)

	else:
		while len(rewards) > 0:
			guess_rewards = rewards + [-1]*(n-len(rewards))
			G = mutiplier.dot(guess_rewards)
			model.update(states[0], actions[0], G)
			states.pop(0)
			actions.pop(0)
			rewards.pop(0)

	return totalreward

def plot_cost_to_go(env, model, num_tiles = 20):
	x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num_tiles)
	y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num_tiles)

	X, Y = np.meshgrid(x,y)
	Z = np.apply_along_axis(lambda _: -np.max(model.predict(_)), 2, np.dstack([X,Y]))

	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(X, Y, Z,
		rstride=1,
		cstride=1,
		cmap='hsv', 
		vmin=-1.0, 
		vmax=1.0)

	ax.set_xlabel('Position')
	ax.set_ylabel('Velocity')
	ax.set_zlabel('Cost To Go Function')

	fig.colorbar(surf)
	plt.show()

def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(totalrewards[max(0,t-100):(t+1)])

	plt.plot(running_avg)
	plt.title('Running Average')
	plt.show()
	
if __name__ == '__main__':
	env = gym.make('MountainCar-v0')
	ft = FeatureTransform(env)
	model = Model(env, ft)
	gamma = 0.99

	if not os.path.exists('Results'):
		os.mkdir('Results')
	os.chdir('Results')
	model.env = wrappers.Monitor(model.env, 'N_Step_Result', force=True)

	N = 300
	totalrewards = np.empty(N)
	for n in range(N):
		eps = 1/(0.1*n+1)
		totalreward = play_one(model, eps, gamma)
		totalrewards[n] = totalreward
		print(f'Episode : {n}, Total Reward: {totalreward}')

	plt.plot(totalrewards)
	plt.title('Total Reward')
	plt.show()

	plot_running_avg(totalrewards)
	plot_cost_to_go(model.env, model)