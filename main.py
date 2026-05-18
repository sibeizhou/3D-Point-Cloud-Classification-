import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import trimesh
import argparse
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np

from dataloader import ModelNetDataset
from model import PointNetClassifier

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for points, labels in train_loader:
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * points.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device, index_to_category):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    num_classes = 10
    class_correct = np.zeros(num_classes, dtype=int).tolist()
    class_total = np.zeros(num_classes, dtype=int).tolist()

    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)

            outputs = model(points)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * points.size(0)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Calculate per-class accuracy
            for label, prediction in zip(labels, predicted):
                class_total[label.item()] += 1
                if label == prediction:
                    class_correct[label.item()] += 1

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print("\nPer-class accuracy:")
    for i in range(num_classes):
        class_name = index_to_category[i]
        class_accuracy = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Class {i} ({class_name}): {class_accuracy:.2f}%")

    return epoch_loss, epoch_acc

def visualize_predictions(model, test_dataset, device, export_folder):
    os.makedirs(export_folder, exist_ok=True)
    model.eval()

    selected_indices = {}
    for idx, (path, label) in enumerate(test_dataset):
        if label not in selected_indices:
            selected_indices[label] = idx
        if len(selected_indices) == 10:
            break

    with torch.no_grad():
        for class_idx, sample_idx in selected_indices.items():
            points, label = test_dataset[sample_idx]
            class_name = test_dataset.return_class(sample_idx)

            mesh_file = test_dataset.data[sample_idx]
            mesh = trimesh.load(mesh_file, force='mesh')
            mesh.export(os.path.join(export_folder, f'{class_name}_mesh.obj'))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(os.path.join(export_folder, f"{class_name}_sample_points.ply"), pcd)

            points = torch.tensor(points).unsqueeze(0).to(device)

            outputs = model(points)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            top3 = probs.argsort()[-3:][::-1]

            print("=" * 50)
            print(f"Sample from Class {class_name}")
            for rank, pred_idx in enumerate(top3):
                print(f"Top {rank+1} prediction: {test_dataset.index_to_category[pred_idx]} ({probs[pred_idx]*100:.2f}%)")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    num_points = args.sample_points

    augments = [
        "rotate",
        "scaling",
        "translation",
        "jitter",
        "dropout"
    ]
    train_dataset = ModelNetDataset(root='ModelNet10', split='train', num_points=num_points, augments=augments)
    test_dataset = ModelNetDataset(root='ModelNet10', split='test', num_points=num_points, augments=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = PointNetClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if not args.skip_train:
        best_acc = 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device, test_dataset.index_to_category)

            print(f"Epoch [{epoch+1}/{num_epochs}]\n"
                f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}\n"
                f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}\n")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'best_model.pth')

        print("Final Best Test Accuracy: {:.4f}".format(best_acc))

    model.load_state_dict(torch.load('best_model.pth'))
    final_loss, final_acc = evaluate(model, test_loader, criterion, device, test_dataset.index_to_category)

    visualize_predictions(model, test_dataset, device, export_folder='result/visualization_exports')
    print("Overall accuracy: {:.2f}%".format(final_acc * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-points', type=int, default=2048, help='Number of sample points')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--skip-train', action='store_true', help='Skip training and only run evaluation + visualization')
    args = parser.parse_args()
    main(args)