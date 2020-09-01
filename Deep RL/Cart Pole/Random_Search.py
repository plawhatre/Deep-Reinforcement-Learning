import gym, os
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

def get_action(s, w):
	return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params):
	observation = env.reset()
	t = 0
	done = False
	while not done:
		t += 1
		action = get_action(observation, params)
		observation, reward, done, info = env.step(action)
	return t 


def play_many_episode(env, T, params):
	episode_lengths = np.empty(T)

	for i in range(T):
		episode_lengths[i] = play_one_episode(env, params)

	avg_length = np.mean(episode_lengths)
	print('Average episode length: ', avg_length)
	return avg_length

def random_search(env):
	episode_lengths = []
	best = 0
	params = None

	for _ in range(100):
		new_params = np.random.random(4)*2-1
		avg_length = play_many_episode(env, 100, new_params)
		episode_lengths.append(avg_length)

		if avg_length >best:
			params = new_params
			best = avg_length

	return episode_lengths, params

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	episode_lengths, params = random_search(env)
	plt.plot(episode_lengths)
	plt.show()

	print('__________Final Play__________')
	if not os.path.exists('Results'):
		os.mkdir('Results')
	os.chdir('Results')
	env = wrappers.Monitor(env, 'Random_Search_Result', force=True)
	avg_length = play_many_episode(env, 100, params)
