
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json


class InferenceHandler:

    def __init__(self, model_name: str, device=torch.device('cpu')) -> None:
        self.device = device
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _translate_m2m100(self, en_text):
        self.tokenizer.src_lang = "en"
        tokens = self.tokenizer(en_text, return_tensors="pt").to(self.device)
        generated_tokens = self.model.generate(
            **tokens, forced_bos_token_id=self.tokenizer.get_lang_id("th"), num_beams=5, do_sample=False)
        th_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)
        return th_text[0]

    def _translate_pipeline(self, en_text):
        translator = pipeline('translation', model=self.model,
                              tokenizer=self.tokenizer, src_lang='eng_Latn', tgt_lang='tha_Thai')
        output = translator(en_text, max_length=400)
        th_text = output[0]['translation_text']
        return th_text

    def translate_text(self, en_text: str):
        if 'm2m100' in self.model_name:
            th_text = self._translate_m2m100(en_text)
        else:
            th_text = self._translate_pipeline(en_text)
        return th_text

    def translate_doc(self, full_en_text: str):
        full_en_text = full_en_text.replace('<|endoftext|>', '')
        full_th_text = []
        for line in full_en_text.split('\n'):
            line = line.strip()
            if line == '':
                full_th_text.append(line.strip())
            else:
                if line.startswith('Rosey:'):
                    if line == 'Rosey:':
                        full_th_text.append('Rosey:')
                    else:
                        full_th_text.append(
                            'Rosey:' + self.translate_text(line.replace('Rosey:', '')))
                else:
                    full_th_text.append(self.translate_text(line))

        full_th_text = '\n'.join(full_th_text)
        return full_th_text + '<|endoftext|>'


"""
model_name can be
# facebook/m2m100_1.2B
# facebook/m2m100-12B-last-ckpt
# facebook/nllb-200-3.3B
"""


def main(text_file: str, model_name="facebook/m2m100_418M", device=torch.device('cpu')):

    with open(text_file) as f:
        entries = [row[1] for row in json.load(f)]

    handler = InferenceHandler(model_name, device=device)

    full_en_text = entries[0]

    print('en_text -> ', full_en_text)
    full_th_text = handler.translate_doc(full_en_text)
    print('th_text -> ', full_th_text)


if __name__ == '__main__':
    main('/home/kunato/language-model-agents/instruction_tuning_dataset_alpha_part1.json')
