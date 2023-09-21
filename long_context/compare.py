import torch
max_position_embeddings=4096
scaling_factor = 8
base=10000
# dimension of each of the head
dim=64
#ntk
for seq_len in range(4096, 4096 * 3, 4096):
    max_seq_len_cached = seq_len
    if seq_len > max_position_embeddings:
        base = base * (
            (scaling_factor * seq_len / max_position_embeddings) - (scaling_factor - 1)
        ) ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    else:
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len_cached, dtype=inv_freq.dtype)

    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    print(emb.sin()[None, None, :, :])
    print(emb.sin()[None, None, :, :].shape)

print("ntk done")

max_position_embeddings=4096
scaling_factor = 8
base=10000
dim=64
#rotary embedding
for seq_len in range(4096, 4096 * 3,  4096):
    max_seq_len_cached = seq_len
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len_cached, dtype=inv_freq.dtype)

    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    #print(emb.cos()[None, None, :, :])
    print(emb.sin()[None, None, :, :])
    print(emb.sin()[None, None, :, :].shape)

print("rotary done")

max_position_embeddings=4096
scaling_factor = 8
base=10000
dim=64
#linear scaled rotary embedding
for seq_len in range(4096, 4096 * 3, 4096):
    max_seq_len_cached  = seq_len
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len_cached, dtype=inv_freq.dtype)
    t = t / 4.0 
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    print(emb.sin()[None, None, :, :])
    print(emb.sin()[None, None, :, :].shape)
print("linear scale rotary done")
