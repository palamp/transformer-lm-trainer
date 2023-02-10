# %%
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from time import sleep
import os
import deepspeed
from pythainlp.util import normalize

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
deepspeed.init_distributed("nccl")


class InferenceHandler:

    def __init__(self, model_name: str, bot_name: str, device=torch.device('cuda:0')) -> None:
        print('loading model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.bot_name = bot_name
        self.device = device

        with deepspeed.OnDevice(dtype=torch.float16, device="cuda:0"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name).half()
        self.model = deepspeed.init_inference(self.model,
                                              mp_size=world_size,
                                              dtype=torch.float16,
                                              replace_method='auto',
                                              replace_with_kernel_inject=True)
        print('loaded')

    def _get_instruction_text(self):
        return ''
        # return f'The following is conversation between an Deeple assistant called {self.bot_name}, and a human user called User. {self.bot_name} is intelligent, knowledgeable, wise and polite.'

    def _get_input_text(self, text: str, input_text: str):
        return text + "User:" + input_text + f"\n{self.bot_name}:"

    def _get_resp_text(self, text: str, input_text: str):
        text = re.sub(r'' + self.bot_name +
                      r'[ \t]*:', f'{self.bot_name}:', text)
        right_side_of_input_text = normalize(
            text).split(normalize(input_text))[1].strip()
        resp = right_side_of_input_text.split(
            normalize(f'{self.bot_name}:'))[1].strip()
        if(resp.startswith(':')):
            return resp[1:]
        return resp

    def run_loop(self):
        text = self._get_instruction_text()
        while True:
            sleep(1)
            input_text = input()
            if input_text == 'RESET':
                print("RESET")
                text = self._get_instruction_text()
                continue
            if input_text == '':
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
            decode_text_resp = self._get_resp_text(decode_text, input_text)

            decode_text_resp = decode_text_resp.split(
                'User')[0].strip().split(f'{self.bot_name}:')[0]
            print('BOT:', decode_text_resp)
            text = text + decode_text_resp


# %%
handler = InferenceHandler(
    'results/040', bot_name='Deeple', device=torch.device('cuda:0'))
handler.run_loop()
