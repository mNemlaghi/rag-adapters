import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from .embedders import AdaptedCohereChromaEmbedder
from .models import ReducedLinearLayer
from .evaluators import OptimizedQAEmbeddingEvaluator
import pandas as pd
from tqdm.auto import tqdm


class TripletsDataset(Dataset):
    def __init__(self, triplets ,emb, maxLength= 2048):
        self.triplets = triplets
        self.emb = emb
        self.maxLength = maxLength

    def __len__(self):
        return len(self.triplets)
    
    
    def __getitem__(self, idx):
        tri = self.triplets[idx]
        encoded = self.emb([tri['question'][:self.maxLength], tri['relevant'][:self.maxLength], tri['distractor'][:self.maxLength]]) 
        question = torch.Tensor(encoded[0])
        relevant = torch.Tensor(encoded[1])
        distractor = torch.Tensor(encoded[2])
        return question, relevant, distractor 
    



class FinalAdapterTrainer:
    def __init__(self, train_triplets, valid_triplets, criteria_fn, emb, 
                 lr=1e-4, weight_decay=1e-3, max_epochs=50, batch_size=8, d_hidden=32, 
                 patience=5, min_delta=1):
        
        #Data
        self.train_triplets = train_triplets
        self.valid_triplets = valid_triplets

        #Tools
        self.initial_embedder = emb
        self.criteria_fn = criteria_fn


        #Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.max_epochs = max_epochs
        self.min_delta = min_delta
        self.scaling_factor = 1000
    


        self.ds = TripletsDataset(train_triplets, self.initial_embedder)
        self.valds = TripletsDataset(valid_triplets, self.initial_embedder)
        
        d = len(emb(["This is a test"])[0])
        self.adapter = ReducedLinearLayer(d, d // d_hidden)

        self.loss_values = []
        self.epoch_val_losses = []
        self.hr_values = []

    def evaluate(self, triplets):
        adapted_embedder = AdaptedCohereChromaEmbedder(self.adapter, self.initial_embedder)
        
        with torch.no_grad():
            adapted_eval = OptimizedQAEmbeddingEvaluator(triplets, adapted_embedder)
            return adapted_eval.evaluate()

    def compute_loss(self, v):
        question, relevant, distractor = v
        anchor, positive, negative = map(self.adapter, (question, relevant, distractor))
        
        return self.scaling_factor*self.criteria_fn(anchor, positive, negative)

    def get_best_score(self, score = "similarity_diff"):
        hr_df = pd.DataFrame(self.hr_values)
        return hr_df.query("split=='validation'")[score].max()

    def train(self):
        self.optimizer = optim.Adam(self.adapter.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_loader = DataLoader(self.ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.valds, batch_size=2*self.batch_size, shuffle=False)

        best_val_loss = float('inf')

        for epoch in range(self.max_epochs):
            self.adapter.train()
            epoch_loss = self._train_epoch(train_loader)
            
            self.adapter.eval()
            epoch_val_loss = self._validate_epoch(val_loader)
            
            self._update_metrics(epoch, epoch_loss, epoch_val_loss)
            
            if self._check_early_stopping(epoch_val_loss, best_val_loss):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            best_val_loss = min(best_val_loss, epoch_val_loss)

    def _train_epoch(self, loader):
        epoch_loss = []
        with tqdm(loader, desc=f"Epoch {len(self.epoch_val_losses) + 1}") as tepoch:
            for v in tepoch:
                loss = self.compute_loss(v)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss.append(loss.item())
                tepoch.set_postfix(loss = sum(epoch_loss) / len(epoch_loss) , val_loss = self.epoch_val_losses[-1] if self.epoch_val_losses else 3000 )
                
        return sum(epoch_loss) / len(epoch_loss)

    def _validate_epoch(self, loader):
        val_losses = [self.compute_loss(v).item() for v in loader]
        return sum(val_losses) / len(val_losses)

    def _update_metrics(self, epoch, train_loss, val_loss):
        self.loss_values.append(train_loss)
        self.epoch_val_losses.append(val_loss)
        
        for split, triplets in [("train", self.train_triplets), ("validation", self.valid_triplets)]:
            hr = self.evaluate(triplets)
            hr.update({"epoch": epoch, "split": split})
            self.hr_values.append(hr)

    def _check_early_stopping(self, current_val_loss, best_val_loss):
        if current_val_loss < best_val_loss - self.min_delta:
            self.epochs_no_improve = 0
            return False
        self.epochs_no_improve += 1
        return self.epochs_no_improve >= self.patience