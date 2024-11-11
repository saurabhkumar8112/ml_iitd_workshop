# train.py
import torch
import torchvision
import wandb
from torch import nn
from torchvision import transforms
from typing import Tuple, Dict

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

def get_data_loaders(batch_size: int = 32, num_workers: int = 2) -> torch.utils.data.DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    return torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

def train_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, int, int]:
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    
    return loss.item(), correct, total

def train_model(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 5,
    log_interval: int = 100
) -> None:
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            loss, batch_correct, batch_total = train_batch(
                model, inputs, labels, optimizer, criterion, device
            )
            
            running_loss += loss
            correct += batch_correct
            total += batch_total
            
            if batch_idx % log_interval == log_interval - 1:
                accuracy = 100. * correct / total
                metrics = {
                    'epoch': epoch,
                    'loss': running_loss/log_interval,
                    'accuracy': accuracy
                }
                log_metrics(metrics, batch_idx)
                running_loss = 0.0
                correct = 0
                total = 0

def log_metrics(metrics: Dict[str, float], batch_idx: int) -> None:
    wandb.log(metrics)
    print(
        f"Epoch: {metrics['epoch']} | "
        f"Batch: {batch_idx+1} | "
        f"Loss: {metrics['loss']:.3f} | "
        f"Accuracy: {metrics['accuracy']:.2f}%"
    )

def save_model(model: nn.Module) -> None:
    # Save model
    artifact = wandb.Artifact(
        name=f"mnist-model-{wandb.run.id}", 
        type="model",
        description="MNIST CNN model"
    )
    
    # Save locally first
    torch.save(model.state_dict(), "model.pth")
    
    # Log to wandb
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)

def main():
    # Initialize wandb with config
    config = {
        "learning_rate": 0.001,
        "momentum": 0.5,
        "epochs": 5
    }
    wandb.init(project="ml-pipeline-workshop", config=config)
    
    # Setup device and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader = get_data_loaders()
    
    # Initialize model, optimizer, and criterion
    model = MNISTNet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config["learning_rate"], 
        momentum=config["momentum"]
    )
    criterion = nn.CrossEntropyLoss()
    
    # Log model architecture
    wandb.watch(model)
    
    # Train model
    train_model(model, trainloader, optimizer, criterion, device, num_epochs=config["epochs"])
    
    # Save model
    save_model(model)

if __name__ == '__main__':
    main()