import torch.nn as nn
import torch

class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.group2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.group3 = nn.Sequential(
            nn.Conv3d(32,64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.group4 = nn.Sequential(
            nn.Conv3d(64,64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.group5 = nn.Sequential(
            nn.Conv3d(64,64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )


  
        self.fc = nn.Sequential(
            nn.Linear(64,1))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        m = nn.AdaptiveMaxPool3d((1, 1, 1))
        out = m(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.squeeze(-1)
        return out