import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import load_data, maybe_makedirs

class NN(nn.Module):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT: Create 3 separate branch networks using nn.Sequential() for the three actions
        # HINT: Use either of the following for weight initialization:
        #         - nn.init.xavier_uniform_()
        #         - nn.init.kaiming_uniform_()
        # HINT: you can call self.modules() to loop through all the layers of the network in the class for initialization






        ########## Your code ends here ##########

    def forward(self, x, u):
        x = x.float()
        u = u.int()

        if u.dim() > 1:
            u = u.view(-1)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network.
        # - x is a (?, |O|) tensor that keeps a batch of observations
        # - u is a (?, 1) tensor (a vector indeed) that keeps the high-level commands (goals) to denote which branch of the network to use 
        # HINT 1: Looping over all data samples may not be the most computationally efficient way of doing branching
        # HINT 2: While implementing this, we found using masks useful. This is not necessarily a requirement though.
        







        ########## Your code ends here ##########


def loss_fn(y_est, y):
    y = y.float()
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations & goals,
    # - y is the actions the expert took for the corresponding batch of observations & goals
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally





    ########## Your code ends here ##########
   

def train_model(data, args):
    """
    Trains a feedforward NN. 
    """

    batch_size = 4096
    x_train = torch.tensor(data['x_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    u_train = torch.tensor(data['u_train'], dtype=torch.long)  # expert commands
    
    in_size = x_train.shape[1]
    out_size = y_train.shape[1]
    
    model = NN(in_size, out_size)
    policy_path = os.path.join("policies", f"{args.scenario.lower()}_{args.goal.lower()}_CoIL.pt")
    if args.restore and os.path.exists(policy_path):
        model.load_state_dict(torch.load(policy_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = TensorDataset(x_train, y_train, u_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        for x_batch, y_batch, u_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            u_batch = u_batch.to(device)
            
            ######### Your code starts here #########

            





            ########## Your code ends here ##########
            count += 1

        
        avg_loss = epoch_loss / count if count > 0 else 0.0
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
    
    torch.save(model.state_dict(), policy_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    # For CoIL, the goal is fixed to all
    args.goal = 'all'
    
    maybe_makedirs("./policies")
    
    data = load_data(args)
    train_model(data, args)