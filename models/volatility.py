import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralVolatitlityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_xh = nn.RNNCell(input_dim, hidden_dim)
        self.rnn_hz = nn.RNNCell(hidden_dim+latent_dim, hidden_dim)
        self.mlp_z_mean = nn.Linear(hidden_dim, latent_dim)
        self.mlp_z_log_var = nn.Linear(hidden_dim, latent_dim)
        self.rnn_zh = nn.RNNCell(latent_dim+input_dim, hidden_dim)
        self.mlp_x_mean = nn.Linear(hidden_dim, input_dim)
        self.mlp_x_log_var = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # We will use teacher forcing
        n = x.shape[0]

        hidden_input = torch.randn(n, self.hidden_dim)
        hidden_latent = torch.randn(n, self.hidden_dim)
        hidden_output = torch.randn(n, self.hidden_dim)
        z = torch.randn(n, self.latent_dim)

        preds = []
        for i in range(n):
            hidden_input = self.rnn_xh(x[i], hidden_input)
            hidden_latent = self.rnn_hz(z, hidden_latent)

            mean_z = self.mlp_z_mean(hidden_latent)
            log_var_z = self.mlp_z_log_var(hidden_latent)
            
            noise_z = torch.randn(n, self.latent_dim)
            z = torch.exp(log_var_z) * noise_z + mean_z

            if i == 0:
                tmp = torch.randn(n, self.input_dim)
            else:
                tmp = x[i-1]
            
            zx = torch.cat((z, tmp), dim=1)
            hidden_output = self.rnn_zh(zx, hidden_output)
            mean_x = self.mlp_x_mean(hidden_output)
            log_var_x = self.mlp_x_mean(hidden_output)

            noise_x = torch.randn(n, self.latent_dim)
            x_pred = torch.exp(log_var_x) * noise_x + mean_x
            preds.append(x_pred)
            
        return preds




