# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from time import sleep
import os
from pythainlp.util import normalize
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class InferenceHandler:

    def __init__(self, model1_name: str, model2_name: str, bot_name1: str, bot_name2: str, device=torch.device('cuda:0')) -> None:
        print('loading model')
        self.tokenizer = AutoTokenizer.from_pretrained(model1_name)
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.bot_name1 = bot_name1
        self.bot_name2 = bot_name2
        self.device = device
        self.model1 = AutoModelForCausalLM.from_pretrained(
            model1_name).half().to(device)

        self.model2 = AutoModelForCausalLM.from_pretrained(
            model2_name).half().to(device)
        print('loaded')

    def _get_input_text_1(self, text: str, input_text: str):
        return text + "User:" + input_text + f"\n{self.bot_name1}:"

    def _get_input_text_2(self, text: str, input_text: str):
        return "โปรดใช้ข้อความนี้ตอบ:" + text + "User:" + input_text + f"\n{self.bot_name2}:"

    def _get_resp_text(self, text: str, input_text: str, botname: str):
        right_side_of_input_text = normalize(
            text).split(normalize(input_text))[1].strip()
        resp = right_side_of_input_text.split(
            normalize(f'{botname}'))[1].strip()
        if(resp.startswith(':')):
            return resp[1:]
        return resp

    def forward_model(self, model, text, input_text, n):
        if n == 1:
            text = self._get_input_text_1(text, input_text)
        else:
            text = self._get_input_text_2(text, input_text)
        inputs = self.tokenizer.encode(
            text, return_tensors="pt").to(self.device)
        outputs = model.generate(inputs,
                                 no_repeat_ngram_size=4,
                                 do_sample=True,
                                 top_p=0.95,
                                 temperature=0.5,
                                 max_new_tokens=128,
                                 top_k=4,
                                 repetition_penalty=1.03,
                                 penalty_alpha=0.6,
                                 )
        decode_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        if n == 1:
            resp = self._get_resp_text(
                decode_text, input_text, botname=self.bot_name1).split('User')[0].strip().split(f'{self.bot_name1}:')[0]
        else:
            resp = self._get_resp_text(
                decode_text, input_text, botname=self.bot_name2).split('User')[0].strip().split(f'{self.bot_name2}:')[0]
        return resp

    @torch.no_grad()
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
            decode_text_resp = self.forward_model(
                self.model1, text, input_text, n=1)
            print('MODEL1:', decode_text_resp)
            decode_text_resp2 = self.forward_model(
                self.model2, decode_text_resp, input_text, n=2)

            print('BOT:', decode_text_resp2)
            text = text + decode_text_resp2


# %%
handler = InferenceHandler(
    'results/025', 'results/036', bot_name1='Rosey', bot_name2="Chip", device=torch.device('cuda:0'))
handler.run_loop()

# %%
