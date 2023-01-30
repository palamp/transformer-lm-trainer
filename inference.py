# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from time import sleep
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print('loading model')
model_name = 'results/001'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# TODO custom model based on v2 need this
# tokenizer.pad_token_id = '1'
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16).to('cuda')

print('loaded')

# %%
text = ''
while(True):
    sleep(1)
    input_text = input()
    if input_text == 'RESET':
        text = ''
        continue
    print('User: ', input_text)
    # Are generic brand crocs manufactured in the same location as name brand crocs?
    text = text + "User:" + input_text + "Rosey:"
    inputs = tokenizer.encode(text, return_tensors="pt").to("cuda")
    with open('input.txt', 'w') as w:
        w.write(text)
    outputs = model.generate(inputs,
                             no_repeat_ngram_size=4,
                             do_sample=True,
                             top_p=0.95,
                             temperature=0.5,
                             max_length=2048,
                             #  top_k=4,
                             repetition_penalty=1.03,
                             penalty_alpha=0.6)
    decode_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print('decode', decode_text)
    decode_text = decode_text.replace(text, '').split('User')[
        0]
    print('BOT', decode_text)
    text = text + decode_text

# %%
