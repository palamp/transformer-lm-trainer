import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM


def _decode_and_remove_newline(tokenizer, tensor):
    if len(tensor.shape) == 2:
        tensor = tensor[0]
    decode_text = tokenizer.decode(
        tensor, skip_special_tokens=True)
    remove_newline_decode_text = ' '.join(
        decode_text.split('\n'))
    return remove_newline_decode_text


def main(model_path, type='seq2seq', device=torch.device('cuda:0'), do_sample=False):
    if type == 'seq2seq':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    print('loaded')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if type == 'seq2seq':
        tokenizer.src_lang = "th"
        tokenizer.tgt_lang = 'th'
    text = "Translate: หากคุณกำลังมองหาตัวเลือกมื้ออาหารที่อร่อยและดีต่อสุขภาพ มาที่ข้าวหมูกรอบ! ข้าวหมูกรอบอบสดใหม่พร้อมน้ำจิ้มซีฟู้ดรับรองว่าถูกใจแน่นอน เหมาะสำหรับเป็นของว่างหรืออาหารกลางวันสำหรับวัยรุ่นที่ต้องการพลังงานตลอดทั้งวัน! #วัยรุ่นกินเพื่อสุขภาพ #หมูกรอบราดซอสซีฟู้ด #อร่อยและมีคุณค่าทางโภชนาการOriginal: If you're looking for a delicious and healthy meal option, then come to ข้าวหมูกรอ🥡! Our freshly-baked crispy pork on rice with seafood sauce is sure to satisfy your cravings. Perfect as a snack or lunch for teens who need more energy throughout the day! #HealthyTeenEating #CrispyPorkOnRiceWithSeafoodSauce #DeliciousAndNutritious"
    # labels = 'Paraphase: '
    if type == 'clm':
        text += 'Paraphase:'
    inputs = tokenizer.encode(text, return_tensors="pt")
    kwargs = {
        'no_repeat_ngram_size': 4,
        'do_sample': True,
        'top_p': 0.5,
        'temperature': 0.5,
        'top_k': 2,
        'repetition_penalty': 1.03,
        'penalty_alpha': 0.6,

    } if do_sample else {}
    output = model.generate(inputs.to(device), max_new_tokens=200,
                            **kwargs

                            )
    output = _decode_and_remove_newline(tokenizer, output)
    print('output text', output)
    output = output.split('Paraphase:')[-1].strip()
    return output


if __name__ == '__main__':
    # main('results/063/checkpoint-8000')
    weight_path = 'results/082/checkpoint-1500'
    # weight_path = 'results/082'
    main(weight_path, type='clm')
