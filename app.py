import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Use the same ImprovedCVAE class from your notebook
class ImprovedCVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ImprovedCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = 10

        # Improved Encoder with BatchNorm and Dropout
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + 1, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # 7x7 -> 4x4 (with padding adjustment)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Flatten(),
        )

        # Correct flattened size calculation
        self.flattened_size = 128 * 3 * 3  # 1152

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Improved Decoder
        self.decoder_input = nn.Linear(latent_dim + self.label_dim, 128 * 3 * 3)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 3, 3)),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # 3x3 -> 6x6
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # 6x6 -> 12x12
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=5, stride=2, padding=1
            ),  # 12x12 -> 25x25
            nn.ReLU(),
            nn.ConvTranspose2d(
                1, 1, kernel_size=4, stride=1, padding=0
            ),  # 25x25 -> 28x28
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels_onehot):
        # Better label conditioning - create label map with same spatial dimensions as input
        label_map = labels_onehot.view(-1, 10, 1, 1).expand(-1, -1, 28, 28)
        # Use first channel of label map (or sum across label dimensions)
        label_channel = (
            torch.sum(
                label_map * torch.arange(10, device=x.device).view(1, 10, 1, 1),
                dim=1,
                keepdim=True,
            )
            / 45.0
        )  # normalize by max sum
        x_cond = torch.cat([x, label_channel], dim=1)

        encoded = self.encoder(x_cond)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)

        z_cond = torch.cat([z, labels_onehot], dim=1)
        dec_input = self.decoder_input(z_cond)
        x_recon = self.decoder(dec_input)
        return x_recon, mu, logvar


@st.cache_resource  # Updated caching decorator
def load_model(path="cvae_mnist.pth"):
    """Load the trained CVAE model"""
    try:
        model = ImprovedCVAE(latent_dim=128)  # Match the training parameters
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(
            f"Model file '{path}' not found. Please make sure the model is trained and saved."
        )
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def generate_images(model, digit, num_samples=5):
    """Generate images for a specific digit"""
    if model is None:
        return None

    # Create labels
    labels = torch.full((num_samples,), digit, dtype=torch.long)
    labels_onehot = nn.functional.one_hot(labels, num_classes=10).float()

    # Sample from latent space
    z = torch.randn(num_samples, model.latent_dim)

    with torch.no_grad():
        # Generate images
        z_cond = torch.cat([z, labels_onehot], dim=1)
        decoder_input = model.decoder_input(z_cond)
        generated = model.decoder(decoder_input)

    return generated.numpy()


def main():
    st.title("MNIST Handwritten Digit Generator")
    st.write(
        "Pick the desired digit and number of samples to generate new handwritten digit images."
    )

    # Load model
    with st.spinner("Loading model..."):
        model = load_model()

    if model is None:
        st.stop()

    # st.success("Model loaded successfully!")

    # Main window controls
    st.header("Generation Controls")

    col1, col2 = st.columns(2)

    with col1:
        digit = st.selectbox(
            "Select digit to generate:", options=list(range(10)), index=5
        )

    with col2:
        num_samples = st.slider(
            "Number of samples:", min_value=1, max_value=10, value=5
        )

    # Generate button
    if st.button("🎲 Generate New Images", type="primary"):
        with st.spinner(f"Generating {num_samples} images of digit {digit}..."):
            images = generate_images(model, digit, num_samples)

        if images is not None:
            st.subheader(f"Generated Images of Digit {digit}")

            # Display images in a grid
            cols = st.columns(min(num_samples, 5))
            for i in range(num_samples):
                col_idx = i % 5
                if i > 0 and col_idx == 0:
                    cols = st.columns(min(num_samples - i, 5))

                with cols[col_idx]:
                    # Convert to PIL Image for better display
                    img_array = images[i].squeeze()
                    img_array = (img_array * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode="L")

                    st.image(img, caption=f"Sample {i+1}", width=120)
        else:
            st.error("Failed to generate images")


if __name__ == "__main__":
    main()
