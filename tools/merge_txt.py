# %%
from typing import List
import random


def clean_up(text: str):
    text = text.replace('&#39;', "'")
    text = text.replace('Deple', "Deeple")
    return text


def main(input: str, ext: str, ext_ratio: float, output: str, split='<|endoftext|>'):
    with open(input) as f:
        data_main = f.read()
        row_main = data_main.split(split)
    with open(ext) as f:
        data_ext = f.read()
        row_ext = data_ext.split(split)
    result = row_main
    for i in range(ext_ratio):
        result = result + row_ext

    random.shuffle(result)
    with open(output, 'w') as w:
        for row in result:
            resp = f'{clean_up(row)}<|endoftext|>'
            w.write(resp)


if __name__ == '__main__':
    main('filelist/v2/inst_v1_th_en_test.txt', 'deeple_translate_ext.txt',
         1, output='filelist/v2/inst_v1_th_en_test_deeple_3.txt')

# %%
