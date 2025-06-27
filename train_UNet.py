import fff.loss as loss
import fff.other_losses.exact_nll
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils as U

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device=='cpu':
    num_workers = 0
    pin_memory = False
else: 
    num_workers = 2
    pin_memory = True

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST(root='./data',  train=True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),

            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,1,kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# ---------- basic 3×3 conv block ----------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ---------- encoder step ----------
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

# ---------- decoder step ----------
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_c=out_c*2, out_c=out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ---------- U-Net 28×28 grayscale ----------
class ReconUNet28_Lite(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        # ---- Encoder ----
        self.e1 = EncoderBlock(in_ch, 16)    # 28×28 → 14×14
        self.e2 = EncoderBlock(16, 32)       # 14×14 → 7×7

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(32, 64)

        # ---- Decoder ----
        self.d1 = DecoderBlock(64, 32)       # 7×7 → 14×14
        self.d2 = DecoderBlock(32, 16)       # 14×14 → 28×28

        # ---- Output layer ----
        self.out_conv = nn.Conv2d(16, in_ch, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        b      = self.bottleneck(p2)
        d1     = self.d1(b, s2)
        d2     = self.d2(d1, s1)
        return self.out_conv(d2)


class LatentDistribution(nn.Module):
    def __init__(self,latent_dim, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

    def get_distribution(self):
        return torch.distributions.Independent(
            torch.distributions.Normal(
                loc=torch.zeros(latent_dim, device=self.device),
                scale=torch.ones(latent_dim, device=self.device)
            ),
            1
        )

    def sample(self, sample_shape=torch.Size()):
        return self.get_distribution().sample(sample_shape)

    def log_prob(self, z):
      if z.dim() != 2 or z.size(1) != 784:        # not yet flat to (B, 784)
        z = z.view(z.size(0), -1)
      return self.get_distribution().log_prob(z)


# wrapping flow model
class UNetFFF(nn.Module):
    def __init__(self, latent_dim, device = 'cpu'):
        # constructor
        super().__init__()
        self.encoder = ReconUNet28_Lite(in_ch=1).to(device)
        self.decoder = ReconUNet28_Lite(in_ch=1).to(device)
        self.latent = LatentDistribution(latent_dim, device)

    def encode(self, x):
        z_img = self.encoder(x) # this is after Unet_1
        return z_img.view(z_img.size(0), -1) # returns a flattened [Batchsize, 784]

    def decode(self, z): # this is going into Unet_2
        z_img = z.view(z.size(0), 1, 28, 28) # reshaped to the img tensor
        return self.decoder(z_img) # recon x

    def log_prob_z(self, z):
        # return self.latent.get_distribution().log_prob(z) # what shape should z be here to be able to calculate this?
        return self.latent.log_prob(z)

    def sample(self, num_samples):
        # decode imgs after drawing samples from latent
        z = self.latent.sample(torch.Size([num_samples]))  # [N,784]
        return self.decode(z)


latent_dim = 28*28
model  = UNetFFF(latent_dim, device)


n_epochs      = 1
learning_rate = 1e-4
k_hutch       = 4          # Hutchinson samples
ema_tau       = 0.999
latent_dim = 28*28

# Store results across betas
beta_train_loss = []
beta_recon_loss = []
beta_nll_loss   = []
loss_history    = []
# beta_surr_loss = []
# beta_logprob_loss = []

for beta in [0.01, 1, 10, 100]:
    print("beta =", beta)

    model = UNetFFF(latent_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    model.sur_base = torch.zeros(1, device=device)

    train_epoch_loss = []
    recon_epoch_loss = []
    nll_epoch_loss   = []
    # surrs_loss = []
    # log_probs_loss = []

    for epoch in range(n_epochs):
        print("epoch =", epoch)
        model.train()
        running_loss = running_recon = running_nll = 0.0

        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()

            # 1. Volume surrogate

            ############################## from FFF package #############################
            sur_out = fff.loss.volume_change_surrogate(
                x, model.encoder, model.decoder,
                hutchinson_samples=k_hutch
            )
            ############################## from FFF package #############################

            d  = 784          # 784 for MNIST

            sur_scaled = sur_out.surrogate / d
            model.sur_base = ema_tau * model.sur_base + (1 - ema_tau) * sur_scaled.mean()
            sur_centered  = sur_scaled - model.sur_base.detach()

            # 2. Reconstruction
            x1 = sur_out.x1
            lrecon = (x - x1).pow(2).flatten(1).sum(-1)/784

            # 3. Latent log prob
            z_flat   = sur_out.z.flatten(1)
            log_prob = model.latent.get_distribution().log_prob(z_flat)
            log_prob = log_prob / 784
            # surrs_loss.append(- sur_centered.mean().item())  # volume surrogate
            # log_probs_loss.append(- log_prob.mean().item())  # log-prob

            # 4. Final losses
            lnll = -log_prob - sur_centered
            loss = (beta * lrecon + lnll).mean()

            loss.backward()
            if epoch == 0:
                for n, p in model.encoder.named_parameters():
                    if p.grad is None:
                        print(f"[WARNING] no grad for {n}")
            U.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accumulate batch losses
            running_loss  += loss.item()
            running_recon += lrecon.mean().item()
            running_nll   += lnll.mean().item()

        scheduler.step()
        loss_history.append({"epoch": epoch, "lr": scheduler.get_last_lr()[0]})

        # Average over batches
        avg_loss  = running_loss / len(dataloader)
        avg_recon = running_recon / len(dataloader)
        avg_nll   = running_nll / len(dataloader)

        train_epoch_loss.append(avg_loss)
        recon_epoch_loss.append(avg_recon)
        nll_epoch_loss.append(avg_nll)

        print(f"[Epoch {epoch}] total {avg_loss:.3f}  recon {avg_recon:.3f}  nll {avg_nll:.3f}")

    beta_train_loss.append(train_epoch_loss)
    beta_recon_loss.append(recon_epoch_loss)
    beta_nll_loss.append(nll_epoch_loss)

epochs = range(len(train_epoch_loss))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_epoch_loss, label='Total Loss')
plt.plot(epochs, recon_epoch_loss, label='Reconstruction Loss')
plt.plot(epochs, nll_epoch_loss, label='NLL Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=300)
plt.close()

# Plot and save learning rate schedule
epochs = [entry["epoch"] for entry in loss_history]
lr_vals = [entry["lr"] for entry in loss_history]

plt.figure(figsize=(7, 4))
plt.plot(epochs, lr_vals, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule over Epochs')
plt.grid(True)
plt.tight_layout()
plt.savefig("lr_schedule.png", dpi=300)
plt.close()