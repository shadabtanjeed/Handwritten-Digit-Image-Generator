import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Paste your CVAE class code here (the exact model definition)
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = 10
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flattened_size = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + self.label_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels_onehot):
        label_map = labels_onehot.view(-1, 10, 1, 1).repeat(1, 1, 28, 28)
        x_cond = torch.cat([x, label_map[:, 0:1, :, :]], dim=1)
        encoded = self.encoder(x_cond)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        z_cond = torch.cat([z, labels_onehot], dim=1)
        dec_input = self.decoder_input(z_cond)
        x_recon = self.decoder(dec_input)
        return x_recon, mu, logvar


@st.cache(allow_output_mutation=True)
def load_model(path="cvae_mnist.pth"):
    model = CVAE()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model


def generate_images(model, digit, num_samples=5):
    label = torch.full((num_samples,), digit, dtype=torch.long)
    label_onehot = nn.functional.one_hot(label, num_classes=10).float()

    z = torch.randn(num_samples, model.latent_dim)
    z_cond = torch.cat([z, label_onehot], dim=1)

    with torch.no_grad():
        decoder_input = model.decoder_input(z_cond)
        generated = model.decoder(decoder_input)
    return generated.numpy()


def main():
    st.title("MNIST Digit Generator with CVAE")

    digit = st.slider("Select digit to generate", 0, 9, 0)

    model = load_model()

    images = generate_images(model, digit)

    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img.squeeze(), width=100, clamp=True)


if __name__ == "__main__":
    main()
