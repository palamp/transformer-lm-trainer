
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch


"""
model_name can be
# facebook/m2m100_1.2B
# facebook/m2m100-12B-last-ckpt
# facebook/nllb-200-3.3B
"""


def main(text_file: str, model_name="facebook/m2m100_418M", device=torch.device('cpu')):

    with open(text_file) as f:
        data = f.read()

    entries = data.split("<|endoftext|>")
    en_text = entries[0]
    print('en_text -> ', en_text)

    if 'm2m100' in model_name:
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_name).to(device)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)

        tokenizer.src_lang = "en"
        tokens = tokenizer(en_text, return_tensors="pt").to(device)
        generated_tokens = model.generate(
            **tokens, forced_bos_token_id=tokenizer.get_lang_id("th"))
        th_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        translator = pipeline('translation', model=model,
                              tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='tha_Thai')
        output = translator(en_text, max_length=400)
        th_text = output[0]['translation_text']
    print('th_text -> ', th_text)


if __name__ == '__main__':
    main('/home/kunato/language-model-agents/inst_v1_test.txt',
         model_name='facebook/nllb-200-distilled-600M')
