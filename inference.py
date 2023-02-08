# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from time import sleep
import os
from pythainlp.util import normalize
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class InferenceHandler:

    def __init__(self, model_name: str, bot_name: str, model_type='clm', device=torch.device('cuda:0')) -> None:
        print('loading model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_type = model_type
        self.model_name = model_name
        self.bot_name = bot_name
        self.device = device
        if model_type == 'clm':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name).to(device)
        elif model_type == 'seq2seq':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name).to(device)
        else:
            raise NotImplementedError()
        print('loaded')

    def _get_input_text(self, text: str, input_text: str):
        if self.model_type == 'clm':
            if 'xglm' in self.model_name:
                return text + "User:" + input_text + f"\n{self.bot_name}:" + self.tokenizer.eos_token
            return text + "User:" + input_text + f"\n{self.bot_name}:"
            # return text + "User: " + input_text
        return text + "User:" + input_text

    def _get_resp_text(self, text: str, input_text: str):
        right_side_of_input_text = normalize(
            text).split(normalize(input_text))[1].strip()
        resp = right_side_of_input_text.split(
            normalize(f'{self.bot_name}'))[1].strip()
        if(resp.startswith(':')):
            return resp[1:]
        return resp

    def run_loop(self):
        text = ''
        while True:
            sleep(1)
            input_text = input()
            if input_text == 'RESET':
                print("RESET")
                text = ''
                continue
            print('User: ', input_text)
            # write a short story about an elf maiden named Julia who goes on an adventure with a warrior named Rallio. The two of them have to go through many trials and tribulations in order for the tale to end happily ever after. Tell the story from Rallio's point of view.
            # Why don't cats and dogs get along?
            # I'd like to watch some comedy movies this weekend. Could you recommend a few good ones from the last 20 years?
            # Are generic brand crocs manufactured in the same location as name brand crocs?
            # if model is seq2seq (remove Rosey:)
            text = self._get_input_text(text, input_text)
            inputs = self.tokenizer.encode(
                text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(inputs,
                                          no_repeat_ngram_size=4,
                                          do_sample=True,
                                          top_p=0.95,
                                          temperature=0.5,
                                          max_new_tokens=128,
                                          top_k=4,
                                          repetition_penalty=1.03,
                                          penalty_alpha=0.6,
                                          #   eos_token_id=self.tokenizer.eos_token_id
                                          )
            decode_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True)
            # print('decode_text', decode_text)

            decode_text_resp = self._get_resp_text(decode_text, input_text)

            # print('decode_text_new', decode_text_resp)
            decode_text_resp = decode_text_resp.split(
                'User')[0].strip().split(f'{self.bot_name}:')[0]
            print('BOT:', decode_text_resp)
            text = text + decode_text_resp


# %%
handler = InferenceHandler(
    'results/023', bot_name='Rosey', device=torch.device('cuda:0'))
handler.run_loop()

# %%
