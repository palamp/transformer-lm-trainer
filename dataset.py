import json
import random
from typing import Optional, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class EOSSplitTextDataset(Dataset):
    def __init__(
        self,
        text_file: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length=280,
        arch="dec_only",
        split_kw: Optional[str] = None,
        override_pad_token=True,
        calc_loss_on_pad=True,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.calc_loss_on_pad = calc_loss_on_pad
        self.tokenizer = tokenizer
        self.arch = arch

        if self.arch == "prefix_lm":
            self.split_kw = split_kw

        if "pythia" in tokenizer.name_or_path or "gpt" in tokenizer.name_or_path:
            print("set pad_token_id 1")
            self.tokenizer.pad_token = "<|padding|>"
        elif "mGPT" in tokenizer.name_or_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif "xglm" in tokenizer.name_or_path:
            print("override_pad_token", override_pad_token)
            if override_pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif "m2m" in tokenizer.name_or_path or "nllb-200" in tokenizer.name_or_path:
            self.tokenizer.src_lang = "th"
            self.tokenizer.tgt_lang = "th"

        print("max_length", self.max_length)
        print("calc_loss_on_pad", self.calc_loss_on_pad)
        print("self.tokenizer.pad_token_id", self.tokenizer.pad_token_id)

        with open(text_file) as f:
            data = f.read()
        entries = data.split("<|endoftext|>")
        self.entries = self._filter_entries(entries)

    # based on rallio code
    def _filter_entries(self, entries):
        fixed = []
        for ent in entries:
            if ent.strip() == "":
                continue
            ent = ent.strip("\n")
            fixed.append(ent)
        results = []
        for ent in tqdm(fixed):
            ent = ent + f"{self.tokenizer.eos_token}{self.tokenizer.eos_token}"
            tokens = self.tokenizer.encode(ent)
            if len(tokens) < self.max_length:
                results.append(ent)
        print("total", len(results))
        return results

    def __len__(self):
        return len(self.entries)

    def _remove_batch(self, data):
        new_data = {}
        for k in data.keys():
            new_data[k] = data[k][0]
        return new_data

    def _get_labels(self, labels_tokens):
        if self.calc_loss_on_pad:
            return labels_tokens
        return torch.where(labels_tokens == self.tokenizer.pad_token_id, -100, labels_tokens)

    def __getitem__(self, idx):
        text = self.entries[idx]
        if self.arch == "dec_only":
            tokens = self.tokenizer.encode_plus(
                text,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            tokens["labels"] = self._get_labels(tokens["input_ids"])
        elif self.arch == "prefix_lm":
            if self.split_kw is not None and self.split_kw in text:
                inputs, labels = text.split(self.split_kw)[:2]
            else:
                split_idx = random.randint(1, len(text) - 1)
                inputs = text[:split_idx]
                labels = text[split_idx:]
            inputs_tokens = self.tokenizer.encode_plus(
                inputs,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            labels_tokens = self.tokenizer.encode_plus(
                labels,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                truncation=True,
                return_attention_mask=False,
            )["input_ids"]
            tokens = {
                "input_ids": inputs_tokens["input_ids"],
                "attention_mask": inputs_tokens["attention_mask"],
            }
            tokens["labels"] = self._get_labels(labels_tokens)
        else:
            raise NotImplementedError()
        # {'input_ids': tensor, 'attention_mask': tensor, 'labels': tensor}
        return self._remove_batch(tokens)


class CloneDataset(Dataset):
    def __init__(self, dataset: Dataset, dataset_ratio=0.01, shuffle=True) -> None:
        super().__init__()
        full_dataset_len = len(dataset)
        print(f"total original dataset len: {full_dataset_len}")
        self.dataset_len = int(full_dataset_len * dataset_ratio)
        if shuffle:
            data_idx = [*range(full_dataset_len)]
            random.shuffle(data_idx)
        else:
            data_idx = [*range(dataset)]

        self.data_idx = data_idx[: self.dataset_len]
        print(f"total len dataset: {self.dataset_len}")
        self.dataset = dataset

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.dataset[self.data_idx[idx]]


class JSONDataset(Dataset):
    def __init__(
        self,
        text_file: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length=280,
        calc_loss_on_pad=True,
        arch="dec_only",
    ) -> None:
        with open(text_file) as f:
            data = json.load(f)
        self.arch = arch
        self.calc_loss_on_pad = calc_loss_on_pad
        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer

        if "m2m" in tokenizer.name_or_path or "nllb-200" in tokenizer.name_or_path:
            self.tokenizer.src_lang = "th"
            self.tokenizer.tgt_lang = "th"

    def _remove_batch(self, data):
        new_data = {}
        for k in data.keys():
            new_data[k] = data[k][0]
        return new_data

    def _get_labels(self, labels_tokens):
        if self.calc_loss_on_pad:
            return labels_tokens
        return torch.where(labels_tokens == self.tokenizer.pad_token_id, -100, labels_tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        inputs = "<s><TH_CLS>" + row["backTH"] + "<EN_CLS>" + row["en"]
        labels = "<PP_CLS>" + row["original"] + "</s></s>"
        if self.arch == "enc_dec":
            inputs_tokens = self.tokenizer.encode_plus(
                inputs,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            labels_tokens = self.tokenizer.encode_plus(
                labels,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                truncation=True,
                return_attention_mask=False,
            )["input_ids"]
            tokens = {
                "input_ids": inputs_tokens["input_ids"],
                "attention_mask": inputs_tokens["attention_mask"],
            }
            tokens["labels"] = self._get_labels(labels_tokens)
        elif self.arch == "dec_only":
            tokens = self.tokenizer.encode_plus(
                inputs + labels,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            tokens["labels"] = self._get_labels(tokens["input_ids"])
        else:
            raise NotImplementedError()
        return self._remove_batch(tokens)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer

    config = OmegaConf.load("config/config_paraphrase_base.yaml")
    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-1.7B")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<TH_CLS>", "<EN_CLS>"], "cls_token": "<PP_CLS>"}
    )
    dataset = JSONDataset("data/train_paraphrase_v2.json", tokenizer, **config.data.get("config", {}))
    # dataset = EOSSplitTextDataset(
    #     '/home/kunato/language-model-agents/inst_v1_test.txt', 'facebook/xglm-1.7B', arch='clm', **config.data.config)
    loader = DataLoader(dataset, shuffle=False)
    print("total", len(dataset))
    histogram = [0] * 501
    for b in tqdm(loader):
        # histogram[torch.count_nonzero(b["attention_mask"]).item()] += 1
        print(b)
        print(dataset.tokenizer.decode(b["input_ids"][0]))
        break

    # histogram = np.array(histogram)
    # plt.bar(np.arange(len(histogram)), histogram)
    # plt.show()
