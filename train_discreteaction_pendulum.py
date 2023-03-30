from agent import Agent
import discreteaction_pendulum
import torch

#TODO add plots and code for averging over multiple training runs

def main():
    batch_size=32
    buffer_length=10000
    LR=1e-3
    gamma=0.95
    target_update=500
    eps_start=0.9
    eps_end=0.05
    eps_decay=1000
    n_episodes=600
#    path='Users/niket/DQN/trained_net.pt'             #to save trained model params

    env = discreteaction_pendulum.Pendulum()

    DQN_agent=Agent(env,batch_size,buffer_length,LR,gamma,target_update)
    DQN_agent.build_net()
    DQN_agent.init_policy(eps_start,eps_end,eps_decay,env)
    DQN_agent.train(n_episodes)
    # torch.save(DQN_agent.policy_net.state_dict(), path)
    print('Complete')

    policy=lambda s: DQN_agent.policy.greedy(torch.tensor(s,dtype=torch.float32).unsqueeze(0),DQN_agent.policy_net)

    env.video(policy, filename='figures/my_discreteaction_pendulum.gif')

if __name__ == '__main__':
    main()



