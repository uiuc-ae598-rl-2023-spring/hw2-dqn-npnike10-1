from agent import Agent
import discreteaction_pendulum
import torch
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from utils import *

def main():

    train_flag=0
    plot_flag=1
    gif_flag=0
    ablation_flag=0
    plot_ablation_flag=0

    n_runs=2

    env = discreteaction_pendulum.Pendulum()

    if train_flag==1:
        avg_data,trained_policy,trained_net=many_train_runs(env,n_runs)
        with open('yy_saved_avg_dictionary.pkl', 'wb') as f:
            pickle.dump((avg_data, trained_policy,trained_net),f)
    else:
        with open('yy_saved_avg_dictionary.pkl', 'rb') as f:
            avg_data,trained_policy,trained_net=pickle.load(f)

    policy=lambda s: trained_policy.greedy(torch.tensor(s,dtype=torch.float32).unsqueeze(0),trained_net)
    state_value=lambda s: torch.max(trained_net(torch.tensor(s,dtype=torch.float32).unsqueeze(0)),1)[0].item()

    if plot_flag==1:
        plot_policy(policy,env)
        plot_mean_return(avg_data,n_episodes)
        plot_state_value(state_value)
        plot_trajectory(policy,env)

    if gif_flag==1:
        env.video(policy, filename='figures/trained_discreteaction_pendulum.gif')
    
    if ablation_flag==1:      # naming convention - y/n+y/n = targetQ yes/no + replay yes/no
        with open('yy_saved_avg_dictionary.pkl', 'rb') as f:
            yy_data,yy_trained_policy,yy_trained_net=pickle.load(f)  

        yn_data,yn_trained_policy,yn_trained_net=many_train_runs(env,n_runs,buffer_length=batch_size)
        with open('yn_saved_avg_dictionary.pkl', 'wb') as f:
            pickle.dump((yn_data,yn_trained_policy,yn_trained_net),f)

        ny_data,ny_trained_policy,ny_trained_net=many_train_runs(env,n_runs,target_update=1)
        with open('ny_saved_avg_dictionary.pkl', 'wb') as f:
            pickle.dump((ny_data,ny_trained_policy,ny_trained_net),f)
        
        nn_data,nn_trained_policy,nn_trained_net=many_train_runs(env,n_runs,target_update=1,buffer_length=batch_size)
        with open('nn_saved_avg_dictionary.pkl', 'wb') as f:
            pickle.dump((nn_data,nn_trained_policy,nn_trained_net),f)
    else:
        with open('yy_saved_avg_dictionary.pkl', 'rb') as f:
            yy_data,yy_trained_policy,yy_trained_net=pickle.load(f)  
        
        with open('yn_saved_avg_dictionary.pkl', 'rb') as f:
            yn_data,yn_trained_policy,yn_trained_net=pickle.load(f)

        with open('ny_saved_avg_dictionary.pkl', 'rb') as f:
            ny_data,ny_trained_policy,ny_trained_net=pickle.load(f)
        
        with open('nn_saved_avg_dictionary.pkl', 'rb') as f:
            nn_data,nn_trained_policy,nn_trained_net=pickle.load(f)

    if plot_ablation_flag==1:
        plot_ablation_mean_return(yy_data,yn_data,ny_data,nn_data)



if __name__ == '__main__':
    main()
