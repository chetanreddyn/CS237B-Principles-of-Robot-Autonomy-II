import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import load_data, maybe_makedirs
import torch.distributions as D

class NN(nn.Module):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT: You can use either of the following for weight initialization:
        #         - nn.init.xavier_uniform_()
        #         - nn.init.kaiming_uniform_()
        self.out_size = out_size
        self.hidden_layer1 = nn.Linear(in_size,8)
        self.hidden_layer2 = nn.Linear(8,4)
        self.output_layer = nn.Linear(4,out_size + out_size*out_size)

        self.activation = 'tanh'
        self.epsilon = 1e-6
        ########## Your code ends here ##########

    def forward(self, x):
        x = x.float()
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network.
        # x is a (?, |O|) tensor that keeps a batch of observations
        if self.activation=='tanh':
            x = self.hidden_layer1(x)
            x = F.tanh(x)
            x = self.hidden_layer2(x)
            x = F.tanh(x)
            x = self.output_layer(x)

        elif self.activation=='sigmoid':
            x = self.hidden_layer1(x)
            x = F.sigmoid(x)
            x = self.hidden_layer2(x)
            x = F.sigmoid(x)
            x = self.output_layer(x)

        elif self.activation=='relu':
            x = self.hidden_layer1(x)
            x = F.relu(x)
            x = self.hidden_layer2(x)
            x = F.relu(x)
            x = self.output_layer(x)

        mu = x[:,:self.out_size]
        A = x[:,self.out_size:].view(-1,self.out_size,self.out_size)
        Sigma = torch.bmm(A,A.transpose(1,2)) + self.epsilon*torch.eye(self.out_size,device=x.device).unsqueeze(0)

        Sigma_flat = Sigma.view(-1,self.out_size*self.out_size)

        dist_params = torch.cat((mu,Sigma_flat),dim=1)
        return dist_params
        ########## Your code ends here ##########


   
def loss_fn(y_est, y):
    """
    Calculate the negative log likelihood loss.
    
    y_est: Output of the network, where the first two columns represent the
           mean vector and the remaining four are covariance parameters.
    y: Target actions taken by the expert.
    """
    y = y.float()
    ######### Your code starts here #########
    # We want to compute the negative log-likelihood loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: You may find some of the following functions useful, but feel free to use your own implementation:
    #       - D.MultivariateNormal()
    #       - torch.diag_embed()
    #       - torch.bmm()
    #       - F.softplus()
    # mu = y_est[:, :2] 
    # A = y_est[:, 2:].view(-1, 2, 2) 
    
    # covariance = torch.bmm(A, A.transpose(1, 2))
    
    # # Add jitter for numerical stability
    # jitter = 1e-6 * torch.eye(2, device=y.device).unsqueeze(0)
    # covariance += jitter
    mu = y_est[:,:2]
    covariance = y_est[:,2:].view(-1,2,2)
    
    # Compute log probability under Multivariate Gaussian
    distribution = D.MultivariateNormal(mu, covariance_matrix=covariance)
    log_probs = distribution.log_prob(y)
    return -log_probs.mean()  # Negative mean log-likelihood
    




    ########## Your code ends here ##########


def train_model(data, args):
    params = {
        'train_batch_size': 4096*32,
    }

    x_train = torch.tensor(data['x_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    in_size = x_train.shape[-1]
    out_size = y_train.shape[-1]

    model = NN(in_size, out_size)
    if args.restore:
        model.load_state_dict(torch.load('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=params['train_batch_size'], shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0.0
        count = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            ######### Your code starts here #########
            # HINT: This section is very similar to the section in train_il.py
            y_est = model(x_batch)
            loss = loss_fn(y_est,y_batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            ########## Your code ends here ##########
            count += 1

        avg_loss = train_loss / count if count > 0 else 0.0
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    torch.save(model.state_dict(), './policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=1e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)
    train_model(data, args)