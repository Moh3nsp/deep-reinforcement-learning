# -*- coding: utf-8 -*-
import gym
from Agent_DQN import Agent
from utils import  plot_learning_curve
import numpy as np




 
    
if __name__ == "__main__":
    
    env=gym.make("LunarLander-v2")
    agent= Agent(gamma=0.99,epsilon=1.0,batch_size=64,n_actions=4,
                eps_end=0.05,input_dims=[8],lr=0.003)
    scores,eps_history=[],[]
    ngames=500
    for i in range(ngames):
        score=0
        done=0
        observation = env.reset()
        while not done:
            action=agent.choose_action(observation)
            #env.render()
            observation_,reward,done,info=env.step(action)
            score+=reward
            agent.store_transition(observation,action,reward,observation_,done)
            
            agent.learn()
            observation=observation_
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-100:])
        print('epsilon ' ,i,
              ' score %.2f' % score,
              ' average score %.2f'%avg_score,
              ' epsilon %.2f' % agent.epsilon)
        
    x=[i+1 for i in range(ngames)]
    plot_learning_curve(x,scores, eps_history,'learning_Curve')
    
        
        
        
        
        
        
        