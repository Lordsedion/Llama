from datasets import load_dataset # type: ignore
import tiktoken # type: ignore
import torch # type: ignore
from torch import nn # type: ignore
from torch.nn import functional as F # type: ignore
import numpy as np # type: ignore
from matplotlib import pyplot as plt # type: ignore
import time
import pandas as pd # type: ignore
from model import LLama

device = torch.device("cuda") if torch.cuda.is_available() == True else torch.device("cpu")
print(f"Device: {device}")

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
vocab_size = enc.n_vocab

config = {
    'batch_size': 32,
    'd_model': 64,
    'n_heads': 10,
    'context_window': 256,
    'vocab_size': vocab_size,
    "epochs":2000,
    "log_interval": 10,
    "n_layers": 10
}

def decode(tokens):
    return [enc.decode_single_token_bytes(token) for token in tokens]


datas = load_dataset('bookcorpus')
length = len(datas["train"])

dataset = torch.tensor([token for i in range(0, length//100) for token in enc.encode(datas["train"][i]["text"])], dtype=torch.int64).to(device)
for j in range(1,35):
    print(f"Processing dataset {j} out of 100 iterations...")
    data = torch.tensor([token for i in range(length//100*j, length//100*(j+1)) for token in enc.encode(datas["train"][i]["text"])], dtype=torch.int64).to(device)
    dataset = torch.cat((dataset, data))
    del data

print(f"Training to be done with {int(dataset.shape[0])} tokens")


def get_batches(data, split, batch_size, context_window, config=config):
    train = data[: int(0.8*len(data))]
    val = data[int(0.8*len(data)): int(0.9*len(data))]
    test = data[int(0.9*len(data)):]
    
    batch_data = train
    
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test
    
    #random starting points
    ix = torch.randint(0, batch_data.size(0) - context_window-1, (batch_size, )).to(device)
    x = torch.stack([batch_data[i: i+context_window] for i in ix]).long().to(device)
    y = torch.stack([batch_data[i+1: i+1+context_window] for i in ix]).long().to(device)
    
    return x,y


@torch.no_grad()
def evaluate_loss(model, config=config):
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            loss = loss.mean()
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out  


def train(model, optimizer, scheduler=None, config=config, print_logs=True):
    losses = []
    start_time = time.time()
    best_val_loss = float("inf")
    count = 0
    for epoch in range(config["epochs"]):
        optimizer.zero_grad()
        
        xs, ys = get_batches(dataset, "train", config["batch_size"], config["context_window"])
        xs = xs.to(device)
        ys = ys.to(device)
        logits, loss = model(xs, targets=ys)
        
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model)
            losses += [x]
            count += 10
            
            if x['val'] < best_val_loss:
                count = 0
                best_val_loss = x['val']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'checkpoint.pth')
            
            if print_logs:
                print(f"Epoch {epoch} | train loss {x['train']:.3f} | val loss {x['val']:.3f} | Val loss hasn't decreased for {count} epochs  | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")
            start_time = time.time()
            
            if count >= 100:
                print("Stopping training...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'last_checkpoint.pth')
                break
        
            if scheduler:
                print("lr: ", scheduler.get_lr())
                
    print("validation loss: ", losses[-1]['val'])
    return pd.DataFrame(losses).plot()


llama = LLama(config).to(device)
llama = nn.DataParallel(llama, device_ids = [0,1])
optimizer = torch.optim.Adam(llama.parameters())

train(llama, optimizer, config=config)
