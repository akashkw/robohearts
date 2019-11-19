import gym

from gymhearts.Hearts import *
from gymhearts.Agent.human import Human
from gymhearts.Agent.randomAI import RandomAI
from time import sleep

NUM_EPISODES = 1
MAX_SCORE = 100

# Delay in seconds
DELAY = 0.01

playersNameList = ['Akash Kwatra', 'Lucas Kabela', 'Peter Stone', 'Scott Niekum']
agent_list = [0, 0, 0, 0]

# Human vs Random
agent_list[0] = Human(playersNameList[0], {'print_info' : True})
agent_list[1] = RandomAI(playersNameList[1], {'print_info': False})
agent_list[2] = RandomAI(playersNameList[2], {'print_info': False})
agent_list[3] = RandomAI(playersNameList[3], {'print_info': False})

# Random play
'''
agent_list[0] = RandomAI(playersNameList[0], {'print_info': True})
agent_list[1] = RandomAI(playersNameList[1], {'print_info': True})
agent_list[2] = RandomAI(playersNameList[2], {'print_info': True})
agent_list[3] = RandomAI(playersNameList[3], {'print_info': True})
'''

# Random play muted
'''
agent_list[0] = RandomAI(playersNameList[0], {'print_info': False})
agent_list[1] = RandomAI(playersNameList[1], {'print_info': False})
agent_list[2] = RandomAI(playersNameList[2], {'print_info': False})
agent_list[3] = RandomAI(playersNameList[3], {'print_info': False})
'''

env = gym.make('Hearts_Card_Game-v0')
env.__init__(playersNameList, MAX_SCORE)

for i_episode in range(NUM_EPISODES):
    
    observation = env.reset()
    
    while True:
        sleep(DELAY)
        env.render()
        
        now_event = observation['event_name']
        IsBroadcast = observation['broadcast']
        action = None
        if IsBroadcast == True:
            for agent in agent_list:
                agent.Do_Action(observation)
        else:
            playName = observation['data']['playerName']
            for agent in agent_list:
                if agent.name == playName:
                    action = agent.Do_Action(observation)

        observation, reward, done, info = env.step(action)

        if reward != None:
            #print('\nreward: {0}\n'.format(reward))
            pass

        if done:
            #print('\nGame Over!!\n')
            break
