import torch
from tqdm.auto import tqdm

def batch_embed(texts, emb_function, batch_size=1000):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        embeddings.extend(emb_function(batch))
    return torch.tensor(embeddings)

def compute_cos(a, b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)).diag()

def compute_similarities(augmented_df, emb, chunk_size=10000, tag = ""):
    device = torch.device("cpu")  # Change to "cuda" if you have a GPU

    q_emb = batch_embed(augmented_df['question'].tolist(), emb).to(device)
    p_emb = batch_embed(augmented_df['relevant'].tolist(), emb).to(device)
    n_emb = batch_embed(augmented_df['distractor'].tolist(), emb).to(device)

    pos_sim = []
    neg_sim = []
    relevant_sim = []

    for i in tqdm(range(0, len(augmented_df), chunk_size), desc="Processing"):
        end = min(i + chunk_size, len(augmented_df))
        q_chunk = q_emb[i:end]
        p_chunk = p_emb[i:end]
        n_chunk = n_emb[i:end]

        pos_sim_chunk = compute_cos(q_chunk, p_chunk)
        neg_sim_chunk = compute_cos(q_chunk, n_chunk)
        relevant_sim_chunk = pos_sim_chunk > neg_sim_chunk

        pos_sim.extend(pos_sim_chunk.cpu().tolist())
        neg_sim.extend(neg_sim_chunk.cpu().tolist())
        relevant_sim.extend(relevant_sim_chunk.cpu().tolist())

    augmented_df[f'{tag}positive_similarity'] = pos_sim
    augmented_df[f'{tag}negative_similarity'] = neg_sim
    augmented_df[f'{tag}relevant_sim'] = relevant_sim

    return augmented_df

def add_points_to_df(augmented_df, tag):
    augmented_df[f'{tag}similarity_diff']=augmented_df[f'{tag}positive_similarity'] - augmented_df[f'{tag}negative_similarity']
    return augmented_df
