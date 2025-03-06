import numpy as np
import torch
import gym
import gym_carlo
import time
import argparse
import os
from train_il import NN
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange", default="intersection")
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()
    
    scenario_name = args.scenario.lower()
    assert scenario_name in scenario_names, '--scenario argument is invalid!'

    # Create the environment without the keyword argument,
    # then set the goal parameter manually.
    env = gym.make(scenario_name + 'Scenario-v0')
    if args.goal.lower() == 'all':
        env.goal = len(goals[scenario_name])
    else:
        env.goal = np.argwhere(np.array(goals[scenario_name]) == args.goal.lower())[0, 0]

    model = NN(obs_sizes[scenario_name], 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    policy_path = os.path.join("policies", f"{args.scenario.lower()}_{args.goal.lower()}_IL.pt")
    model.load_state_dict(torch.load(policy_path, map_location=device))
    
    episode_number = 10 if args.visualize else 100
    success_counter = 0
    env.T = 200 * env.dt - env.dt / 2.  # Run for at most 200*dt = 20 seconds

    for _ in range(episode_number):
        env.seed(int(np.random.rand() * 1e6))
        obs = env.reset()
        done = False
        if args.visualize:
            env.render()
        while not done:
            t = time.time()
            # Convert observation to a torch tensor and send to device
            obs_tensor = torch.tensor(np.array(obs).reshape(1, -1), dtype=torch.float32).to(device)
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy().reshape(-1)
            obs, _, done, _ = env.step(action)
            if args.visualize:
                env.render()
                # Wait to maintain a 2x simulation speed
                while time.time() - t < env.dt / 2:
                    pass
            if done:
                env.close()
                if args.visualize:
                    time.sleep(1)
                if hasattr(env, 'target_reached') and env.target_reached:
                    success_counter += 1

    if not args.visualize:
        print('Success Rate =', float(success_counter) / episode_number)