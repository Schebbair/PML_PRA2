import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import lbforaging
import os
from iql import IQL
from cql import CQL

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip #for recording videos

CONFIG = {
    "gamma": 0.95,
    "total_eps": 100000,      
    "lr": 0.05,               
    "init_epsilon": 1.0,
    "eval_epsilon": 0.05,
    "video_folder": "videos_lbf"
}

os.makedirs(CONFIG["video_folder"], exist_ok=True)

#converts the observation in a tuple of integers
class DiscreteWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        new_obs = []
        for agent_obs in obs:
            discretized = (np.array(agent_obs) * 10).astype(int)
            new_obs.append(tuple(discretized))

        return tuple(new_obs)

def run_experiment(env_id, algorithm_name, config):

    env = gym.make(env_id, render_mode="rgb_array", disable_env_checker=True) #disable to avoid error in render_mode
    env = DiscreteWrapper(env)
    
    if algorithm_name == "IQL":
        agent = IQL(
            num_agents=env.unwrapped.n_agents,
            action_spaces=env.action_space,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=config["init_epsilon"]
        )
    elif algorithm_name == "CQL":
        agent = CQL(
            num_agents=env.unwrapped.n_agents,
            action_spaces=env.action_space,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=config["init_epsilon"]
        )
    
    returns = []
    max_steps = config["total_eps"]
    
    for episode in range(config["total_eps"]):
        obss, _ = env.reset()
        done = False
        episodic_reward = np.zeros(env.unwrapped.n_agents)
        
        agent.schedule_hyperparameters(episode, max_steps)
        
        while not done:
            actions = agent.act(obss)
            n_obss, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            
            agent.learn(obss, actions, rewards, n_obss, done)
            
            obss = n_obss
            episodic_reward += rewards
            
        returns.append(np.sum(episodic_reward))
        
        if episode % 1000 == 0:
            print(f"Episode {episode}: Total Return = {np.sum(episodic_reward):.2f}")

    env.close()
    return returns, agent

def record_video(env_id, agent, algorithm_name, config):

    env = gym.make(env_id, render_mode="rgb_array", disable_env_checker=True)
    env = DiscreteWrapper(env)
    
    obss, _ = env.reset()
    done = False
    
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    frames = []
    steps = 0
    max_video_steps = 200 #just in case
    
    try:
        while not done and steps < max_video_steps:

            frame = env.render()
            
            #if the render returns a bool, force the unwrapped mode
            if isinstance(frame, bool) or frame is None:
                frame = env.unwrapped.render(mode='rgb_array')
            
            #check if image is valid
            if isinstance(frame, np.ndarray):
                frames.append(frame)
            
            actions = agent.act(obss)
            obss, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            steps += 1
        
        #save video
        if len(frames) > 0:
            clip = ImageSequenceClip(frames, fps=10)
            output_file = f"{config['video_folder']}/{algorithm_name}_{env_id}.mp4"
            clip.write_videofile(output_file, logger=None)

        else:
            print("aaa")
            
    except Exception as e:
        print(f"Error con el video: {e}")
    finally:
        agent.epsilon = old_epsilon
        env.close()

if __name__ == "__main__":

    scenarios = ["Foraging-5x5-2p-1f-v3","Foraging-5x5-2p-1f-coop-v3"]
    
    results = {}

    for env_id in scenarios:

        #IQL
        iql_returns, iql_agent = run_experiment(env_id, "IQL", CONFIG)
        record_video(env_id, iql_agent, "IQL", CONFIG)
        
        #CQL
        cql_returns, cql_agent = run_experiment(env_id, "CQL", CONFIG)
        record_video(env_id, cql_agent, "CQL", CONFIG)
        
        results[env_id] = {"IQL": iql_returns, "CQL": cql_returns}

    #PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    def moving_average(a, n=500):
        if len(a) < n: return a
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    for i, env_id in enumerate(scenarios):
        ax = axes[i]
        for algo in ["IQL", "CQL"]:
            data = results[env_id][algo]
            smoothed = moving_average(data, n=500)
            ax.plot(smoothed, label=algo)
        
        ax.set_title(f"Environment: {env_id}")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Total Return")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("lbf_comparison.png")