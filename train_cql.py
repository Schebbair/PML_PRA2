import copy
import random

import gymnasium as gym
import numpy as np

from cql import CQL
import matplotlib.pyplot as plt
from matrix_game import create_pd_game

#same config to better comparison
CONFIG = {
    "seed": 0,
    "gamma": 0.99,
    "total_eps": 20000,
    "ep_length": 1,
    "eval_freq": 400,
    "lr": 0.05,
    "init_epsilon": 0.9,
    "eval_epsilon": 0.05,
}

#same structure as train_iql but simplified (it doesn't do the first type of plot)
def train(env, config):
    agent = CQL(
        num_agents=env.n_agents,
        action_spaces=env.action_space,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        epsilon=config["init_epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]
    
    eval_returns = []

    for eps_num in range(config["total_eps"]):
        obss, _ = env.reset()
        done = False
        
        while not done:
            agent.schedule_hyperparameters(step_counter, max_steps)
            actions = agent.act(obss)
            n_obss, rewards, done, _, _ = env.step(actions)
            
            #train
            agent.learn(obss, actions, rewards, n_obss, done)
            
            step_counter += 1
            obss = n_obss

        #evaluate
        if eps_num % config["eval_freq"] == 0:
            eval_agent = agent 
            old_epsilon = agent.epsilon
            agent.epsilon = config["eval_epsilon"]
            
            total_r = 0
            n_eval_eps = 100
            for _ in range(n_eval_eps):
                e_obss, _ = env.reset()
                e_actions = agent.act(e_obss)
                _, e_rewards, _, _, _ = env.step(e_actions)
                total_r += sum(e_rewards) #reward sum of both
            
            avg_return = total_r / n_eval_eps
            eval_returns.append(avg_return)
            agent.epsilon = old_epsilon
            
            print(f"Episode {eps_num}: Joint Return = {avg_return:.2f}")

    return eval_returns, agent

if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    env = create_pd_game()
    returns, agent = train(env, CONFIG)

    #just the second plot
    plt.plot(returns)
    plt.title("CQL Training - Joint Return")
    plt.xlabel("Evaluations")
    plt.ylabel("Sum of Rewards (Agent 1 + Agent 2)")
    plt.axhline(y=-2, color='r', linestyle='--', label="Optimal Coop (-1-1=-2)")
    plt.axhline(y=-6, color='k', linestyle='--', label="Nash Equilibrium (-3-3=-6)")
    plt.legend()
    plt.savefig("cql_results.png")
    plt.show()
    
    print("\n Final Q Table:")
    actions_map = ["C,C", "C,D", "D,C", "D,D"]
    for idx in range(4):
        val = agent.q_table[str(((0,0), idx))]
        print(f"Action {actions_map[idx]}: {val:.2f}")
