import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    #the list of the text mapping
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    #the list of the intent mapping
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    #print(intent2idx)

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    #使用SeqClsDataset來包裝data
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    # dev dataset don't shuffle because yu can not change the label,like test dataset
    train_loader = DataLoader(datasets["train"],batch_size=args.batch_size,shuffle=True,collate_fn=datasets["train"].collate_fn)
    dev_loader = DataLoader(datasets["eval"],batch_size=args.batch_size,shuffle=False,collate_fn=datasets["eval"].collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    #model = nn.RNN(len(train_loader.dataset),len(dev_loader.dataset))
    model = nn.RNN(args.batch_size,args.hidden_size)

    # TODO: init optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    print(train_loader)
    #trange:產生進度條
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        h = torch.rand((1,args.hidden_size))
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.train()
        for i,data in enumerate(train_loader):
            print(data["text"])
            print(data["text"].shape)
            optimizer.zero_grad()
            output,hidden = model(data["text"],h)
            batch_loss = loss(output,data["intent"])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()
        
        model.eval()
        with torch.no_grad():
            for i,data in enumerate(dev_loader):
                val_pred = model(data["text"],12)
                batch_loss = loss(val_pred,data["indent"])

                val_loss += batch_loss.item()
        
    # TODO: Inference on test set
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
