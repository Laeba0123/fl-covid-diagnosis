import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageListDataset(Dataset):
    def __init__(self, txtfile, transform=None):
        self.items = []
        with open(txtfile, 'r') as f:
            for line in f:
                p, cls = line.strip().split('\t')
                self.items.append((p, int(cls)))
        self.transform = transform

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        p, cls = self.items[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, cls

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, train_txt, test_txt, device="cpu"):
        self.cid = cid
        self.device = device
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        self.train_loader = DataLoader(
            ImageListDataset(train_txt, transform=transform),
            batch_size=32,
            shuffle=True
        )
        self.test_loader = DataLoader(
            ImageListDataset(test_txt, transform=transform),
            batch_size=64
        )
        
        from model import SmallCNN
        self.model = SmallCNN(num_classes=4).to(device)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Client {cid} initialized with {len(self.train_loader.dataset)} training samples")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
    
    def fit(self, parameters, config):
       self.set_parameters(parameters)
       optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
    
       self.model.train()
       total_samples = 0
       total_loss = 0.0
    
       for epoch in range(config.get("local_epochs", 1)):
        epoch_loss = 0.0
        for batch_idx, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            
            # Optional: Print every 10 batches
            if batch_idx % 10 == 0:
                print(f"Client {self.cid} | Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        epoch_loss /= len(self.train_loader.dataset)
        print(f"Client {self.cid} | Epoch {epoch+1} Completed | Avg Loss: {epoch_loss:.4f}")
        total_loss += epoch_loss
    
       avg_loss = total_loss / config.get("local_epochs", 1)
       print(f"Client {self.cid} | Training Complete | Avg Loss: {avg_loss:.4f}")
       return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, correct = 0.0, 0
        self.model.eval()
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss += self.criterion(output, y).item() * y.size(0)
                correct += (output.argmax(1) == y).sum().item()
        
        accuracy = correct / len(self.test_loader.dataset)
        return float(loss / len(self.test_loader.dataset)), len(self.test_loader.dataset), {"accuracy": accuracy}

    def get_properties(self, config):
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--test_txt", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    client = FLClient(args.cid, args.train_txt, args.test_txt, args.device)
    
    # Test client initialization
    params = client.get_parameters({})
    loss, num, metrics = client.evaluate(params, {})
    logger.info(f"Initial evaluation - loss: {loss:.4f}, accuracy: {metrics['accuracy']:.4f}")