from typing import Any, List, Dict
import torch
import numpy as np
import torch.nn.functional as F

class OptimizedQAEmbeddingEvaluator:
    def __init__(self, triplets: List[Dict[str, str]], embedder: Any, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.triplets = triplets
        self.embedder = embedder
        self.device = device

    def batch_embed(self, texts: List[str], batch_size: int = 1024) -> torch.Tensor:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedder(batch)
            embeddings.extend(batch_embeddings)
        return torch.tensor(embeddings, device=self.device)

    def evaluate_batch(self, batch: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        questions = [item['question'] for item in batch]
        relevants = [item['relevant'] for item in batch]
        distractors = [item['distractor'] for item in batch]
        
        q_emb = self.batch_embed(questions)
        r_emb = self.batch_embed(relevants)
        d_emb = self.batch_embed(distractors)

        # Vectorized similarity computation
        relevant_sims = F.cosine_similarity(q_emb, r_emb)
        distractor_sims = F.cosine_similarity(q_emb, d_emb)

        pairwise_correct = (relevant_sims > distractor_sims).float()
        mrr = 1.0 / (2 - pairwise_correct)  # 1 if correct, 0.5 if not
        similarity_diff = relevant_sims - distractor_sims

        return {
            'pairwise_correct': pairwise_correct.cpu().numpy(),
            'mrr': mrr.cpu().numpy(),
            'relevant_similarity': relevant_sims.cpu().numpy(),
            'similarity_diff': similarity_diff.cpu().numpy()
        }

    def evaluate(self) -> Dict[str, float]:
        batch_size = 64  # Adjust based on your GPU memory and performance needs
        results = []

        for i in range(0, len(self.triplets), batch_size):
            batch = self.triplets[i:i+batch_size]
            results.append(self.evaluate_batch(batch))

        # Combine results
        combined = {k: np.concatenate([r[k] for r in results]) for k in results[0].keys()}
        return {k: np.mean(v) for k, v in combined.items()}