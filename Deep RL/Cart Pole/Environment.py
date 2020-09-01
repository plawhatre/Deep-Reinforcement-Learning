import gym

#create cart pole environment
env = gym.make('CartPole-v0')

#start state:{cart position, cart velocity, pole angle, pole velocity}
env.reset()

#observation space
box = env.observation_space
print('_____________Obervation Space_____________')
print('\tContains: ', box.contains)
print('\tSample: ',box.sample())
print('\tShape: ',box.shape)
print('\tHigh: ', box.high)
print('\tLow: ', box.low)
print('\tFrom Jsonable: ',box.from_jsonable)
print('\tTo Jsonable: ',box.to_jsonable)

#action space
print('_____________Action Space_____________')
AS = env.action_space
print('\tNumber: : ', AS.n)
print('\tContains: ', AS.contains)
print('\tSample: ', AS.sample())
print('\tShape: ', AS.shape)
print('\tFrom Jsonable: ', AS.from_jsonable)
print('\tTo Jsonable: ', AS.to_jsonable)

#Episode
print('_____________Play an Episode_____________')
done = False
t = 0
while not done:
	t += 1
	env.render()
	observation, reward, done, info = env.step(AS.sample())
env.close()
print(f'\tEpisode finished: {done}, in {t} steps')
print('\tEpisode observation: ', observation)
print('\tEpisode reward: ', reward)



