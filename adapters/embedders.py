from chromadb import Documents, EmbeddingFunction, Embeddings
import json
import hashlib
from typing import List, Dict, Tuple
from functools import lru_cache
import torch
import torch.nn.functional as F

import json

import hashlib

class CohereChromaEmbedder(EmbeddingFunction):
    def __init__(self, session, modelId="cohere.embed-english-v3", initial_cache: Dict[str, List[float]] = {}, lru_cache_size: int = 100000):
        self.client = session.client('bedrock-runtime')
        self.modelId = modelId
        self.cache = initial_cache
        self.invocation_counter = 0
        self.embed = lru_cache(maxsize=lru_cache_size)(self._embed)
        
    def __call__(self, input: Documents) -> Embeddings:
        return self.embed(tuple(input))  # Convert to tuple for hashability
    
    def _embed(self, texts: Tuple[str], batch_size: int = 50) -> List[List[float]]:
        results: List[Tuple[int, List[float]]] = []
        queue: List[Tuple[int, str]] = []

        for i, text in enumerate(texts):
            hash_key = self.hasher(text)
            if hash_key in self.cache:
                results.append((i, self.cache[hash_key]))
            else:
                queue.append((i, text))

            if len(queue) >= batch_size:
                self._process_queue(queue, results)
                queue = []

        if queue:
            self._process_queue(queue, results)

        return [embedding for _, embedding in  sorted(results, key=lambda x: x[0])]

    def _process_queue(self, queue: List[Tuple[int, str]], results: List[Tuple[int, List[float]]]) -> None:
        indices, texts = zip(*queue)
        embeddings = self._invoke_api(texts)
        
        for idx, text, embedding in zip(indices, texts, embeddings):
            hash_key = self.hasher(text)
            self.cache[hash_key] = embedding
            results.append((idx, embedding))

    def _invoke_api(self, texts: List[str]) -> List[List[float]]:
        inputs = {"texts": texts, "input_type": "search_document"}
        response = self.client.invoke_model(modelId=self.modelId, body=json.dumps(inputs))
        self.invocation_counter += 1
        output = json.loads(response['body'].read().decode('utf8'))
        return output['embeddings']

    @staticmethod
    def hasher(s: str) -> str:
        return hashlib.sha256(s.encode('utf8')).hexdigest()


class AdaptedCohereChromaEmbedder:
    def __init__(self,  adapter,  initial_embedder):
        self.adapter = adapter
        self.initial_embedder = initial_embedder
        
    def __call__(self, input: Documents)->Embeddings:
        non_adapted = self.initial_embedder(input)
        x = torch.Tensor(non_adapted)
        y = self.adapter(x)
        
        normalized_y = F.normalize(y, p=2, dim=1)

        return normalized_y.tolist()