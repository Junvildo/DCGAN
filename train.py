import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import *
from data import *
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Arguments users used when running command lines
    parser.add_argument('--train-path', type=str, help='Where training data is located')
    args = parser.parse_args()

    # Hyperparameters etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE_G = 0.0002
    LEARNING_RATE_D = 0.0002
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    NOISE_DIM = 100
    NUM_EPOCHS = 10
    FEATURES_DISC = 64
    FEATURES_GEN = 64

    transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageDataset(data_dir = args.train_path,
                                        transform = transforms)

    train_dataloader = DataLoader(dataset = train_dataset,
                                    batch_size = 128,
                                    shuffle = True)

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    weights_init(gen)
    weights_init(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.9))
    criterion = nn.BCELoss()


    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(train_dataloader):
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_dataloader)} \
                    Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )
