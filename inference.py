# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from time import sleep
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print('loading model')
model_name = 'results/005'
device = torch.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_name)
# TODO custom model based on v2 need this
# tokenizer.pad_token_id = '1'
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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
    # write a short story about an elf maiden named Julia who goes on an adventure with a warrior named Rallio. The two of them have to go through many trials and tribulations in order for the tale to end happily ever after. Tell the story from Rallio's point of view.
    # Why don't cats and dogs get along?
    # I'd like to watch some comedy movies this weekend. Could you recommend a few good ones from the last 20 years?
    # Are generic brand crocs manufactured in the same location as name brand crocs?
    text = text + "User:" + input_text + "Rosey:"
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
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
    decode_text = decode_text.replace(text, '').split('User')[0]
    print('BOT', decode_text)
    text = text + decode_text

# %%
