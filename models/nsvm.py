import torch
import torch.nn as nn

class InfNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.mlp_z_mean = nn.Linear(hidden_dim, latent_dim)
        self.mlp_z_logvar = nn.Linear(hidden_dim, latent_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h0 = torch.zeros(1, batch_size, self.hidden_dim)

        out, _ = self.rnn(x, h0)

        out = out.reshape(-1, self.hidden_dim)

        logvar_z = self.mlp_z_logvar(out)
        mean_z = self.mlp_z_mean(out)

        noise = torch.randn(batch_size*seq_len, self.latent_dim)
        z = noise * torch.exp(logvar_z) + mean_z
        z = z.reshape(batch_size, seq_len, self.latent_dim)

        return z, mean_z, logvar_z


class GenNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim + latent_dim, hidden_dim, batch_first=True)
        self.mlp_x_mean = nn.Linear(hidden_dim, 1)
        self.mlp_x_logvar = nn.Linear(hidden_dim, 1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
    
    def forward(self, x, z):
        batch_size, seq_len, _ = x.shape
        h0 = torch.zeros(1, batch_size, self.hidden_dim)

        out, _ = self.rnn(torch.cat([x,z], dim=-1), h0)

        out = out.reshape(-1, self.hidden_dim)
        logvar_x = self.mlp_x_logvar(out)
        mean_x = self.mlp_x_mean(out)

        noise = torch.randn(batch_size * seq_len,1)
        x = noise * torch.exp(logvar_x) + mean_x
        x = x.reshape(batch_size, seq_len, 1)

        return x, mean_x, logvar_x


inf = InfNet(10, 16, 5)
gen = GenNet(10, 16, 5)

optim_inf = torch.optim.Adam(inf.parameters())
optim_gen = torch.optim.Adam(gen.parameters())

for _ in tqdm.trange(200):
    optim_gen.zero_grad()
    optim_inf.zero_grad()

    z, _, _ = inf(X_train_t)
    _, mu_x, logvar_x = gen(X_train_t, z)
    loss = torch.mean((mu_x[:-1] - Xs[0, 1:, -1:]) ** 2 / torch.exp(logvar_x[:-1]) + logvar_x[:-1])
    loss.backward(retain_graph=True)

    optim_gen.step()
    optim_inf.step()