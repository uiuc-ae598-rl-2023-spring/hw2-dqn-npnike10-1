import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev
import seaborn as sns
from agent import Agent
from tqdm import tqdm 

batch_size=32
buffer_length=100000
LR=0.00025
gamma=0.95
target_update=500
eps_start=0.9
eps_end=0.05
eps_decay=1000
n_episodes=600

def plot_ablation_mean_return(yyd,ynd,nyd,nnd,n_episodes=n_episodes):
    yy_mean=get_mean(yyd,'returns')
    yy_stdev=get_stdev(yyd,'returns')

    yn_mean=get_mean(ynd,'returns')
    yn_stdev=get_stdev(ynd,'returns')

    ny_mean=get_mean(nyd,'returns')
    ny_stdev=get_stdev(nyd,'returns')

    nn_mean=get_mean(nnd,'returns')
    nn_stdev=get_stdev(nnd,'returns')

    ms=[(yy_mean,yy_stdev),(yn_mean,yn_stdev),(ny_mean,ny_stdev),(nn_mean,nn_stdev)]

    plt.figure()
    clrs = sns.color_palette("husl", 1)
    with sns.axes_style("darkgrid"):
        eps = list(range(n_episodes))
        for pair in ms:
            plt.plot(eps, pair[0], c=clrs[0])
            plt.fill_between(eps, pair[0]-pair[1], pair[0]+pair[1] ,alpha=0.3, facecolor=clrs[0])
        plt.ylim(0,25)
        plt.legend(('Standard DQN','With target net without replay', 'Without target net with replay','Without target net without replay'))
        plt.show()


def many_train_runs(env,n_runs,batch_size=batch_size,buffer_length=buffer_length,LR=LR,gamma=gamma,target_update=target_update,eps_start=eps_start,eps_end=eps_end,eps_decay=eps_decay,n_episodes=n_episodes):
    avg_data={'returns':[],'avg_eps_rewards':[],'rewards':[],'real_times':[]}
    for i in tqdm(range(n_runs),desc="Training Run"):
        DQN_agent=Agent(env,batch_size,buffer_length,LR,gamma,target_update)
        DQN_agent.build_net()
        DQN_agent.init_policy(eps_start,eps_end,eps_decay,env)
        data=DQN_agent.train(n_episodes)
        avg_data['returns'].append(data['return'])
        avg_data['avg_eps_rewards'].append(data['avg_eps_reward'])
        avg_data['rewards'].append(data['reward'])
        avg_data['real_times'].append(data['real_time'])
    return avg_data, DQN_agent.policy,DQN_agent.policy_net

def plot_policy(policy,env):
    theta = np.linspace(-np.pi, np.pi, 500)
    thetadot = np.linspace(-15, 15, 500)
    X, Y = np.meshgrid(theta,thetadot)
    tau=np.empty_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            tau[i,j]=env._a_to_u(policy((X[i,j],Y[i,j])))
    plt.imshow(tau, extent=[-np.pi, np.pi, -15, 15], origin='lower',
           cmap='viridis',aspect='auto')
    plt.colorbar()
    plt.savefig('figures/trained_policy_example.png')
    plt.show()

def plot_state_value(state_value):
    theta = np.linspace(-np.pi, np.pi, 500)
    thetadot = np.linspace(-15, 15, 500)
    X, Y = np.meshgrid(theta,thetadot)
    value=np.empty_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            value[i,j]=state_value((X[i,j],Y[i,j]))
    plt.imshow(value, extent=[-np.pi, np.pi, -15, 15], origin='lower',
           cmap='viridis',aspect='auto')
    plt.colorbar()
    plt.savefig('figures/trained_state_value_example.png')
    plt.show()

def plot_mean_return(data,n_episodes):
    mean=get_mean(data,'returns')
    stdev=get_stdev(data,'returns')
    plt.figure()
    clrs = sns.color_palette("husl", 1)
    with sns.axes_style("darkgrid"):
        eps = list(range(n_episodes))
        plt.plot(eps, mean, c=clrs[0],label="Mean return")
        plt.fill_between(eps, mean-stdev, mean+stdev ,alpha=0.3, facecolor=clrs[0], label=r'$1-\sigma$ deviation')
        plt.ylim(0,25)
        plt.ylabel('Return')
        plt.xlabel('Number of Episodes')
    plt.savefig('figures/mean_return.png')
    plt.show()

def plot_trajectory(policy,env):
    s = env.reset()
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = policy(s)
        (s_new, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s_new)
        data['a'].append(a)
        data['r'].append(r)
        s=s_new

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = [env._a_to_u(a) for a in data['a']]

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(data['t'][:-1], data['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('figures/trained_agent_trajectory_example.png')
    plt.show()

def plot_avg_reward(data):  
    plt.plot(data['avg_eps_rewards'][-1])

def get_mean(dict,key):
    return np.array([mean(value) for value in list(zip(*dict[key]))])

def get_stdev(dict,key):
    return np.array([stdev(value) for value in list(zip(*dict[key]))])