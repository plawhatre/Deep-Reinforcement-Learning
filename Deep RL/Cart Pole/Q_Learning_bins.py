import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import os, sys
from gym import wrappers
from datetime import datetime

def build_state(features):
	return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
	return np.digitize([value], bins= bins)[0]

class FeatureTransform:
	def __init__(self):
		'''
		we can sample observations from the environment and plot histogram.
		The bins should be selected in such a manner so that the hist if flat
		'''  
		self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
		self.cart_velocity_bins = np.linspace(-2, 2, 9)
		self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
		self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

	def transform(self, obervation):
		cart_pos, cart_vel, pole_angle, pole_vel = obervation
		return build_state([
			to_bin(cart_pos, self.cart_position_bins),
			to_bin(cart_vel, self.cart_velocity_bins),
			to_bin(pole_angle, self.pole_angle_bins),
			to_bin(pole_vel, self.pole_velocity_bins),
			])

class Model:
	def __init__(self, env, feature_transformer):
		self.env = env
		self.feature_transformer =feature_transformer

		num_states = 10**env.observation_space.shape[0]
		num_actions = env.action_space.n
		self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

	def predict(self, s):
		x = self.feature_transformer.transform(s)
		return self.Q[x]

	def update(self, s, a, G):
		x = self.feature_transformer.transform(s)
		self.Q[x,a] += 10e-3*(G - self.Q[x,a])

	def sample_action(self, s, eps):
		#this implements epsilon greedy
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			p = self.predict(s)
			return np.argmax(p)

def play_one(model, eps, gamma):
	observation = env.reset()
	done = False
	total_reward = 0
	itr = 0
	while not done:
		action = model.sample_action(observation, eps)
		prev_observation = observation
		observation, reward, done, _ = env.step(action)

		total_reward += reward
		if done and itr < 50:
			reward = -300

		#update the model
		G = reward + gamma*np.max(model.predict(observation))
		model.update(prev_observation, action, G)

		itr += 1
	return total_reward

def plt_running_avg(totalreward):
	N = len(totalreward)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = totalreward[max(0,t-100):(t+1)].mean()
	plt.plot(running_avg)
	plt.title('Running Average')
	plt.show()

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	ft = FeatureTransform()
	model = Model(env, ft)
	gamma = 0.9

	if not os.path.exists('Results'):
		os.mkdir('Results')
	os.chdir('Results')
	env = wrappers.Monitor(env, 'Q_Learning_bins_Result', force=True)	

	N = 10000
	totalrewards =  np.empty(N)
	for n in range(N):
		eps = 1/ np.sqrt(n+1)
		total_reward = play_one(model, eps, gamma)
		totalrewards[n] = total_reward
		if n%100 == 0:
			print(f"Episode: {n}, Reward: {total_reward}, Epsilon: {eps}")

	plt.plot(totalrewards)
	plt.title('Reward')
	plt.show()

	plt_running_avg(totalrewards)










