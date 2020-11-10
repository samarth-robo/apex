import torch


def mlp(sizes, activation, output_activation=torch.nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i+1]), act()]
    return torch.nn.Sequential(*layers)

class MLPCategoricalActor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, x):
        pi = torch.distributions.Categorical(logits=self.logits_net(x))
        return pi

class MLPGaussianActor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, std):
        super().__init__()
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.std = torch.tensor(std, dtype=torch.float32)

    def forward(self, x):
        pi = torch.distributions.Normal(loc=self.mu_net(x), scale=self.std)
        return pi
    
    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pi = self.forward(x)
        a = pi.sample().detach()
        logp = pi.log_prob(a).detach().sum(axis=-1)
        return a.numpy(), logp.numpy()

class MLPCritic(torch.nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, x):
        return self.v_net(x)

