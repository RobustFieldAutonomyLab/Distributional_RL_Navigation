import torch
import torch.nn as nn
import numpy as np
import os
import json

class IQN(nn.Module):
    '''implicit quantile network model'''
    def __init__(self, state_size, action_size, layer_size, seed, device="cpu"):
        super(IQN, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.seed_id = seed
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.K = 32 # for action selection
        self.N = 8 # num tau
        self.n_cos = 64 # embedding dimension
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(self.device)
        self.head = nn.Linear(self.state_size, layer_size)
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.hidden_layer = nn.Linear(layer_size, layer_size)
        self.hidden_layer_2 = nn.Linear(layer_size, layer_size)
        self.output_layer = nn.Linear(layer_size, action_size)
        #weight_init([self.head_1, self.ff_1])


    def calc_cos(self, batch_size, n_tau=8, cvar=1.0):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).to(self.device).unsqueeze(-1) # (batch_size, n_tau, 1) for broadcast
        #print(taus)

        # distorted quantile sampling
        taus = taus * cvar

        cos = torch.cos(taus * self.pis)
        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus


    def forward(self, inputs, num_tau=8, cvar=1.0):
        """
        Quantile calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = inputs.shape[0]
        
        x = torch.relu(self.head(inputs))
        cos, taus = self.calc_cos(batch_size, num_tau, cvar) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)
        
        x = torch.relu(self.hidden_layer(x))
        x = torch.relu(self.hidden_layer_2(x))
        out = self.output_layer(x)
        return out.view(batch_size, num_tau, self.action_size), taus


    def get_qvals(self, inputs, cvar):
        quantiles, _ = self.forward(inputs=inputs, num_tau=self.K, cvar=cvar)
        qvals = quantiles.mean(dim=1)
        return qvals

    def get_constructor_parameters(self):       
        return dict(state_size=self.state_size, \
                    action_size=self.action_size, \
                    layer_size=self.layer_size, \
                    seed=self.seed_id)

    def save(self,directory):
        # torch.save({"state_dict": self.state_dict(), "constructor_params": self.get_constructor_parameters()}, path)
        # torch.save(self.state_dict(), path)

        # save network parameters
        torch.save(self.state_dict(),os.path.join(directory,"network_params.pth"))
        
        # save constructor parameters
        with open(os.path.join(directory,"constructor_params.json"),mode="w") as constructor_f:
            json.dump(self.get_constructor_parameters(),constructor_f)

    @classmethod
    def load(cls,directory,device="cpu"):
        
        # load network parameters
        model_params = torch.load(os.path.join(directory,"network_params.pth"),
                                  map_location=device)

        # load constructor parameters
        with open(os.path.join(directory,"constructor_params.json"), mode="r") as constructor_f:
            constructor_params = json.load(constructor_f)
            constructor_params["device"] = device

        model = cls(**constructor_params)
        model.load_state_dict(model_params)
        model.to(device)

        return model



