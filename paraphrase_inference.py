from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def _decode_and_remove_newline(tokenizer, tensor):
    if len(tensor.shape) == 2:
        tensor = tensor[0]
    decode_text = tokenizer.decode(
        tensor, skip_special_tokens=True)
    remove_newline_decode_text = ' '.join(
        decode_text.split('\n'))
    return remove_newline_decode_text


def main(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.src_lang = "th"
    tokenizer.tgt_lang = 'th'
    text = "Translate: หากคุณกำลังมองหาตัวเลือกมื้ออาหารที่อร่อยและดีต่อสุขภาพ มาที่ข้าวหมูกรอบ! ข้าวหมูกรอบอบสดใหม่พร้อมน้ำจิ้มซีฟู้ดรับรองว่าถูกใจแน่นอน เหมาะสำหรับเป็นของว่างหรืออาหารกลางวันสำหรับวัยรุ่นที่ต้องการพลังงานตลอดทั้งวัน! #วัยรุ่นกินเพื่อสุขภาพ #หมูกรอบราดซอสซีฟู้ด #อร่อยและมีคุณค่าทางโภชนาการOriginal: If you're looking for a delicious and healthy meal option, then come to ข้าวหมูกรอ🥡! Our freshly-baked crispy pork on rice with seafood sauce is sure to satisfy your cravings. Perfect as a snack or lunch for teens who need more energy throughout the day! #HealthyTeenEating #CrispyPorkOnRiceWithSeafoodSauce #DeliciousAndNutritious"
    # labels = 'Paraphase: '
    inputs = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(inputs, max_length=200)
    output = _decode_and_remove_newline(tokenizer, output)
    output = output.replace('Paraphase: ', '')
    return output


if __name__ == '__main__':
    main('results/063/checkpoint-8000')
