import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

        self.register_buffer('identity', torch.eye(k).view(1, k * k))

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x + self.identity.repeat(batch_size, 1)
        x = x.view(batch_size, self.k, self.k)
        return x

class SimpleAttention(nn.Module):
    def __init__(self, num_features):
        super(SimpleAttention, self).__init__()
        self.fc = nn.Linear(num_features, num_features)

    def forward(self, x):
        alpha = torch.sigmoid(self.fc(x))
        return x * alpha

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNetClassifier, self).__init__()

        self.spatial_transform = TNet(k=3)

        # Shared MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)

        # self.feature_transform = TNet(k=64)

        self.fc1 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.2)

        self.attention = SimpleAttention(num_features=128)

        self.fc2 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)

        # Input Transform
        trans = self.spatial_transform(x)
        x = torch.bmm(trans, x)

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))

        # trans_feat = self.feature_transform(x)
        # x = torch.bmm(trans_feat, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))

        # Attention
        x = self.attention(x)

        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x

def test_model():
    dummy_input = torch.rand(8, 2048, 3)

    model = PointNetClassifier(num_classes=10)
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    predicted_class = torch.argmax(output, dim=1)
    print("Predicted class indices:", predicted_class)


if __name__ == "__main__":
    test_model()
