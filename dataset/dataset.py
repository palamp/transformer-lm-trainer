from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional
import random
from tqdm import tqdm


class EOSSplitTextDataset(Dataset):

    def __init__(self, text_file: str, tokenizer_name: str, max_length=280, arch='clm', split_kw: Optional[str] = None, override_pad_token=True, calc_loss_on_pad=True) -> None:
        super().__init__()
        self.max_length = max_length
        self.calc_loss_on_pad = calc_loss_on_pad
        self.override_pad_token = override_pad_token
        print('calc_loss_on_pad', self.calc_loss_on_pad)
        print('override_pad_token', self.override_pad_token)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if 'pythia' in tokenizer_name or 'gpt' in tokenizer_name:
            print('set pad_token_id 1')
            self.tokenizer.pad_token = '<|padding|>'
        elif "mGPT" in tokenizer_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif "xglm" in tokenizer_name:
            if self.override_pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        print('self.tokenizer.pad_token_id', self.tokenizer.pad_token_id)
        self.arch = arch
        with open(text_file) as f:
            data = f.read()
        if self.arch == 'prefix_lm':
            self.split_kw = split_kw

        entries = data.split("<|endoftext|>")
        self.entries = self._filter_entries(entries)

    # based on rallio code
    def _filter_entries(self, entries):
        fixed = []
        for ent in entries:
            if ent.strip() == '':
                continue
            ent = ent.strip('\n')
            fixed.append(ent)
        results = []
        for ent in tqdm(fixed):
            ent = ent + \
                f'{self.tokenizer.eos_token}{self.tokenizer.eos_token}'
            tokens = self.tokenizer.encode(ent)
            if len(tokens) < self.max_length:
                results.append(ent)
        print('total', len(results))
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
        if self.arch == 'clm':
            tokens = self.tokenizer.encode_plus(
                text, padding='max_length', max_length=self.max_length, return_tensors='pt', return_attention_mask=True, truncation=True)
            tokens['labels'] = self._get_labels(tokens['input_ids'])
        elif self.arch == 'prefix_lm':
            if self.split_kw is not None and self.split_kw in text:
                inputs, labels = text.split(self.split_kw)[:2]
            else:
                split_idx = random.randint(1, len(text) - 1)
                inputs = text[:split_idx]
                labels = text[split_idx:]
            inputs_tokens = self.tokenizer.encode_plus(
                inputs, padding='max_length', max_length=self.max_length, return_tensors='pt', return_attention_mask=True, truncation=True)
            labels_tokens = self.tokenizer.encode_plus(
                labels, padding='max_length', max_length=self.max_length, return_tensors='pt', truncation=True, return_attention_mask=False)['input_ids']
            tokens = {'input_ids': inputs_tokens['input_ids'],
                      'attention_mask': inputs_tokens['attention_mask']}
            tokens['labels'] = self._get_labels(labels_tokens)
        else:
            raise NotImplementedError()
        # {'input_ids': tensor, 'attention_mask': tensor, 'labels': tensor}
        return self._remove_batch(tokens)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import torch
    from omegaconf import OmegaConf
    config = OmegaConf.load('config/config_clm.yaml')
    dataset = EOSSplitTextDataset(
        '/home/kunato/language-model-agents/inst_v1_test.txt', 'facebook/xglm-1.7B', arch='clm', **config.data.config)
    loader = DataLoader(dataset, shuffle=False)
    print('total', len(dataset))
    for b in loader:
        print(b)
        print(b['input_ids'].sum())
        print(torch.count_nonzero(b['attention_mask']))
        break
