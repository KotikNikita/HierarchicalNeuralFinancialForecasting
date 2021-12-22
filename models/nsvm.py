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

class NSVM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.gen = GenNet(input_dim, hidden_dim, latent_dim)
        self.inf = InfNet(input_dim, hidden_dim, latent_dim)
        self.beta = 0.1

    def forward(self, x, noise=True):
        z, mu_z, logvar_z = self.inf(x)
        if noise == False:
            z = mu_z.unsqueeze(0)
        x, mu_x, logvar_x = self.gen(x, z)
        return mu_x, logvar_x, mu_z, logvar_z

    def loss(self, x, y):
        mu_x, logvar_x, mu_z, logvar_z = self(x)
        recon_loss = torch.mean((mu_x[:-1] - y) ** 2 / 2 / torch.exp(2 * logvar_x[:-1]) + logvar_x[:-1])
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar_z - mu_z ** 2 - logvar_z.exp(), dim = 1), dim = 0)
        loss = recon_loss + self.beta * kld_loss
        return loss

    def predict(self, x, noise=True):
        with torch.no_grad():
            y_hat = self(x, noise)
        return y_hat
    
    def multi_predict(self, x, n, noise=True):
        x_new = torch.clone(x)
        for _ in range(n):
            x_pred = self.predict(x_new[-xwid:])
            x_new = torch.cat([x, x_pred])
        return x_new