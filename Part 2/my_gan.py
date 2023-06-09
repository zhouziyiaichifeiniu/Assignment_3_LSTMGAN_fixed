import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128 * 7 * 7)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        z = self.net(z)
        z = z.reshape(-1, 128, 7, 7)
        z = self.conv(z)
        return z


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        # self.net = nn.Sequential(
        #     nn.Linear(784, 512),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
        )

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.linear = nn.Linear(8192, 1)

    def forward(self, img):
        # return discriminator score for img
        # output = self.convnet(img)
        output = self.net(img)
        output = torch.flatten(output).reshape(-1, 8192)
        output = torch.sigmoid(self.linear(output))
        return output


def generate_noise():
    mean = 0
    var = 1
    size = (64, 100)
    noise = torch.randn(size) * var + mean
    # print(noise)
    # print(noise.shape)
    if torch.cuda.is_available():
        noise = noise.cuda()
    return noise


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    real_label = torch.ones(64).cuda()
    fake_label = torch.zeros(64).cuda()
    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        real_label = real_label.cuda()
        fake_label = fake_label.cuda()
    loss = nn.BCELoss()

    for epoch in range(args.n_epochs):
        print(epoch)
        for i, (imgs, _) in enumerate(dataloader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
            # imgs = imgs.reshape(imgs.shape[0], -1)
            if imgs.shape[0] < 64:
                continue;
            # print(imgs.shape)
            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            # 生成的假图片
            fake_image = generator(generate_noise())
            # fake_image = fake_image.view(64, 1, 28, 28)
            # 假图片的评分
            fake_result = discriminator(fake_image)
            ge_loss = loss(fake_result, real_label.unsqueeze(1))
            ge_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            if epoch % 2 != 0:
                optimizer_D.zero_grad()
                real_result = discriminator(imgs)
                fake_image = generator(generate_noise())
                # fake_image = fake_image.view(64, 1, 28, 28)
                fake_result = discriminator(fake_image)
                l1 = loss(real_result, real_label.unsqueeze(1))
                l2 = loss(fake_result, fake_label.unsqueeze(1))
                # 同时判断真假图片的loss
                l = (l1 + l2) / 2
                l.backward(retain_graph=True)
                optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                # save_image(gen_imgs[:25],
                #            'images/{}.png'.format(batches_done),
                #            nrow=5, normalize=True, value_range=(-1,1))
                save_image(fake_image[:25], 'images/{}.png'.format(batches_done), nrow=5, normalize=True,
                           value_range=(-1, 1))
                torch.save(generator.state_dict(), './model/G_{}.pth'.format(batches_done))


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5),
                                                (0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=2000,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
