from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Optional
import random


class EOSSplitTextDataset(Dataset):

    def __init__(self, text_file: str, tokenizer: PreTrainedTokenizer, max_length=280, arch='clm', split_kw: Optional[str] = None) -> None:
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.arch = arch
        with open(text_file) as f:
            data = f.read()

        if self.arch == 'prefix_lm':
            self.split_kw = split_kw

        entries = data.split("<|endoftext|>")
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def _remove_batch(self, data):
        new_data = {}
        for k in data.keys():
            new_data[k] = data[k][0]
        return new_data

    def __getitem__(self, idx):
        text = self.entries[idx]
        if self.arch == 'clm':
            tokens = self.tokenizer.encode_plus(
                text, padding='max_length', max_length=self.max_length, return_tensors='pt', return_attention_mask=True, truncation=True)
            tokens['labels'] = tokens['input_ids']
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
            tokens['labels'] = labels_tokens
        else:
            raise NotImplementedError()
        # {'input_ids': tensor, 'attention_mask': tensor, 'labels': tensor}
        return self._remove_batch(tokens)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    tokz = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
    tokz.pad_token_id = 1
    dataset = EOSSplitTextDataset(
        '/home/kunato/language-model-agents/inst_clean_v3.txt', tokz, arch='prefix_lm')
    loader = DataLoader(dataset)
    for b in loader:
        print(b)
        break
