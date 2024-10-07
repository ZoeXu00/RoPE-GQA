"""
Trains a character-level language model.
"""

import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import pickle
import wandb
# -----------------------------------------------------------------------------

def get_config():

    C = CN()
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    C.data = CharDataset.get_default_config()

    C.model = GPT.get_default_config()
    C.model.n_layer=6
    C.model.n_query_head=6
    C.model.n_kv_head=6
    C.model.n_embd=192
    C.model.rope = False

    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 16
        C.tokenizer="default"
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = [self.stoi[s] for s in data]

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        dix = self.data[idx:idx + self.config.block_size + 1]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    wandb_api = "318ce0a3be337c2a1d962e4113b5fff247b502a1"
    wandb.login(key=wandb_api)
    configs = {"max_iters": 600, "sequence_length": 16}
    wandb.init(
        project = "10623_HW1",
        name = "without RoPe",
        config = configs
    )   

    config = get_config()
    config.merge_from_args(sys.argv[1:])
    set_seed(config.system.seed)

    text = open('input.txt', 'r').read()
    train_dataset = CharDataset(config.data, text)

    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    print(config)
    model = GPT(config.model)
    
    if config.model.pretrained_folder!=None:
        assert os.path.normpath(os.path.abspath(config.model.pretrained_folder)) != os.path.normpath(os.path.abspath(config.system.work_dir)), "pretrained folder cannot be same as current folder. Change the folder name of your pretrained model or current directory using flags"
        model.load_pretrained(config.model.pretrained_folder)
    
    setup_logging(config)

    trainer = Trainer(config.trainer, model, train_dataset)

    train_losses = []
    attn_times = []
    attn_mem = []

    def batch_end_callback(trainer):
        if trainer.iter_num % 1 == 0:
            train_losses.append(trainer.loss.item())
            attn_times.append(trainer.attn_times*1000)
            if trainer.device=="cuda":
                print(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f};attn_times {trainer.attn_times*1000:.2f}ms;mem_consumed {trainer.memory_consumed/(1024*1024):.2f}MB")
                attn_mem.append(trainer.memory_consumed/(1024*1024))
            else:
                print(f"iter_dt {trainer.iter_dt:.2f}s; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f};attn_times {trainer.attn_times*1000:.2f}ms;mem_consumed - not available on CPU")

            wandb.log({f'Loss': trainer.loss.item()}, step = trainer.iter_num)

        if (trainer.iter_num + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                context = "O God, O God!"
                encoded_context = [train_dataset.stoi[s] for s in context]
                x = torch.tensor(encoded_context, dtype=torch.long)[None,...].to(trainer.device)
                y, attn_time = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)
                y = y[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
                print(f"Attention computation took {attn_time*1000:.2f}ms to run for {config.data.block_size} seq length")
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print("saving loss and attention logs")
            with open(os.path.join(config.system.work_dir, 'train_losses.json'), 'w') as f:
                json.dump(train_losses, f, ensure_ascii=False, indent=4)
            with open(os.path.join(config.system.work_dir, 'attention_computation_time.json'), 'w') as f:
                json.dump(attn_times, f, ensure_ascii=False, indent=4)
            with open(os.path.join(config.system.work_dir, 'attention_computation_memory.json'), 'w') as f:
                json.dump(attn_mem, f, ensure_ascii=False, indent=4)
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
