import torch
import torch.nn as nn

class GAFMaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 24 -> 12
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 6 -> 12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 12 -> 24
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    

class GAFDemandClassifier(nn.Module):
    def __init__(self, encoder, num_classes=4, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits
    

class GAFWeatherHistoryFusionClassifier(nn.Module):
    def __init__(self, encoder, history=24, num_weather_features=3, num_classes=4, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        weather_input_dim = history * num_weather_features

        self.weather_mlp = nn.Sequential(
            nn.Linear(weather_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_img, x_weather):
        z_img = self.encoder(x_img)
        z_img = self.pool(z_img).flatten(1)

        x_weather = x_weather.flatten(1)
        z_weather = self.weather_mlp(x_weather)

        z = torch.cat([z_img, z_weather], dim=1)
        return self.classifier(z)