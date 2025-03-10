import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import load_data, maybe_makedirs
import os
import time

import gym
import gym_carlo
from utils import *

import pdb
import wandb

class NN(nn.Module):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT: Use either of the following for weight initialization:
        #         - nn.init.xavier_uniform_
        #         - nn.init.kaiming_uniform_
        
        
        self.hidden_layer1 = nn.Linear(in_features = in_size, out_features = 32)
        self.hidden_layer2 = nn.Linear(in_features = 32, out_features = 64)
        self.hidden_layer3 = nn.Linear(in_features = 64, out_features = 32)
        self.output_layer =  nn.Linear(in_features = 32,out_features=out_size)
        self.activation = 'tanh' # 'tanh' or 'softsign' or 'sigmoid'

        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.xavier_uniform_(self.hidden_layer2.weight)
        nn.init.xavier_uniform_(self.hidden_layer3.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        ########## Your code ends here ##########

    def forward(self, x):
        x = x.float()
        ######### Your code starts here #########
        # Perform a forward-pass of the network. 
        # x is a (?,|O|) tensor that keeps a batch of observations
        
        # x has a shape of )batch_size, obs_dim)
        if self.activation=='tanh':
            x = self.hidden_layer1(x)
            x = F.tanh(x)
            x = self.hidden_layer2(x)
            x = F.tanh(x)
            x = self.hidden_layer3(x)
            x = F.tanh(x)
            x = self.output_layer(x)

        elif self.activation=='sigmoid':
            x = self.hidden_layer1(x)
            x = F.sigmoid(x)
            x = self.hidden_layer2(x)
            x = F.sigmoid(x)
            x = self.hidden_layer3(x)
            x = F.sigmoid(x)
            x = self.output_layer(x)

        elif self.activation=='softsign':
            x = self.hidden_layer1(x)
            x = F.softsign(x)
            x = self.hidden_layer2(x)
            x = F.softsign(x)
            x = self.hidden_layer3(x)
            x = F.softsign(x)
            x = self.output_layer(x)

        elif self.activation=='relu':
            x = self.hidden_layer1(x)
            x = F.relu(x)
            x = self.hidden_layer2(x)
            x = F.relu(x)
            x = self.hidden_layer3(x)
            x = F.relu(x)
            x = self.output_layer(x)
        return x
        ########## Your code ends here ##########


def loss_fn(y_est, y,Q):
    y = y.float()
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally
    
    # Shape of y is (batch_size,2)
    
    # Q is a 2x2 Weight matrix
    # error_vec = (y-y_est)@Q@(y-y_est).T
    # loss = torch.mean(error_vec)


    return F.mse_loss(y_est,y)
    ########## Your code ends here ##########
    

def nn_train(data, args,wandb_config_dict):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': wandb_config_dict['train_batch_size'],
    }


    x_train = torch.tensor(data["x_train"])
    y_train = torch.tensor(data["y_train"])
    in_size = x_train.shape[-1]
    out_size = y_train.shape[-1]

    scenario_name = args.scenario.lower()
    env = gym.make(scenario_name + 'Scenario-v0')

    # env = gym.make(scenario_name + 'Scenario-v0')
    if args.goal.lower() == 'all':
        env.goal = len(goals[scenario_name])
    else:
        env.goal = np.argwhere(np.array(goals[scenario_name]) == args.goal.lower())[0, 0]

    model = NN(in_size, out_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ",device)
    model.to(device)

    Q = np.array([[wandb_config_dict['Q_steering'],0],
                  [0,wandb_config_dict['Q_throttle']]])
    Q = torch.tensor(Q,device=device).float()

    policy_path = os.path.join("policies", f"{args.scenario.lower()}_{args.goal.lower()}_IL.pt")
    if args.restore and os.path.exists(policy_path):
        model.load_state_dict(torch.load(policy_path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataset = TensorDataset(x_train[:], y_train[:])


    
    train_loader = DataLoader(dataset, batch_size=params['train_batch_size'], shuffle=True)
    # pdb.set_trace()

    wandb_config_dict['device'] = device.type
    if wandb_config_dict["wandb_log"]:
        wandb.init(project="CS237B Imitation Learning",
                config=wandb_config_dict
               )

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device) # Shape of (batch_size,obs_dim)
            y_batch = y_batch.to(device) # Shape of (batch_size,act_dim)


            ######### Your code starts here #########
            # We want to compute the loss between y_est and y where
            # - y_est is the output of the network for a batch of observations,
            # - y_batch is the actions the expert took for the corresponding batch of observations
            # At the end your code should return the scalar loss value.
            y_est = model(x_batch)
            loss = loss_fn(y_est,y_batch,Q)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            # print("Loss: {:5f}".format(loss))
            # print(y_est)    




            ########## Your code ends here ##########
            count += 1
        

        avg_loss = train_loss / count if count > 0 else 0.0

        log_dict = {"Epoch":epoch+1,
                    "Average Epoch Loss":avg_loss}
        
        if wandb_config_dict["wandb_log"]:
            wandb.log(log_dict,step = epoch+1)
            
        if epoch%wandb_config_dict['print_every'] == 0:
            duration = time.time() - start_time
            start_time = time.time()

            # model.eval()
            # success_rate = nn_test(model,args,device,env)
            # print("Duration: {:2f} | Epoch {} | Loss: {:.6f} | Test Success Rate: {:.2f}".format(duration,epoch+1,avg_loss,success_rate))
            print("Duration: {:2f} | Epoch {} | Loss: {:.6f}".format(duration,epoch+1,avg_loss))


            # if wandb_config_dict["wandb_log"]:
            #     wandb.log({"Success Rate":success_rate},step=epoch+1)

    if wandb_config_dict["wandb_log"]:
        wandb.finish()


    torch.save(model.state_dict(), policy_path)

def nn_test(model,args,device,env):

    '''
    The function is used to return the success rate which is logged into wandb
    '''


    episode_number = 100 
    success_counter = 0
    env.T = 200 * env.dt - env.dt / 2.  # Run for at most 200*dt = 20 seconds

    for _ in range(episode_number):
        env.seed(int(np.random.rand() * 1e6))
        obs = env.reset()
        done = False
        # if args.visualize:
        #     env.render()
        while not done:
            t = time.time()
            # Convert observation to a torch tensor and send to device
            obs_tensor = torch.tensor(np.array(obs).reshape(1, -1), dtype=torch.float32).to(device)
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy().reshape(-1)
            obs, _, done, _ = env.step(action)
            # if args.visualize:
            #     env.render()
            #     # Wait to maintain a 2x simulation speed
            #     while time.time() - t < env.dt / 2:
            #         pass
            if done:
                env.close()
                # if args.visualize:
                #     time.sleep(1)
                if hasattr(env, 'target_reached') and env.target_reached:
                    success_counter += 1

    success_rate = float(success_counter/episode_number)
    return success_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange", default="intersection")
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--wandb_log", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    data = load_data(args)

    wandb_config_dict = {'lr':args.lr,
                         'epochs':args.epochs,
                         'n_hidden_layer1':32,
                         'n_hidden_layer2':64,
                         'n_hidden_layer3':32,
                         'train_batch_size':4096,
                         'wandb_log':args.wandb_log,
                         'Q_steering':2.0,
                         'Q_throttle':1.0,
                         'print_every':10,
                         'eval_episode_number':100}

    nn_train(data, args,wandb_config_dict)

