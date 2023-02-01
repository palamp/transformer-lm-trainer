```
<!-- original -->
defaultdict(<class 'int'>, {'dahoas': 32718, 'anthrop_redteam': 6521, 'anthrop_helpful': 6359, 'synth_code': 3444, 'synth_qa': 15329, 'conala': 2759})
```


```
<!-- remove -->
defaultdict(<class 'int'>, {'anthrop_helpful': 1447, 'anthrop_redteam': 10, 'dahoas': 22, 'synth_qa': 18, 'synth_code': 5})
```

```
<!-- move & generate new -->
defaultdict(<class 'int'>, {'anthrop_helpful': 3001, 'dahoas': 36676, 'synth_code': 3158, 'synth_qa': 23526, 'anthrop_redteam': 5636, 'conala': 2707})
```

```
what is in dahoas?
Dahoas/synthetic-instruct-gptj-pairwise -> 33143
Dahoas/synthetic_prompt_responses -> 6681
```

```
What include in new?
gsm8k
2764 not gsm8k ? another synth_qa?
```


#### Problem of THAI
```
User:  โรงแรมไหนดีในกรุงเทพ
BOT: Rosey เป็นโรงแรมที่ดีในกรีสซึ่งมาพร้อมกับบริการอาหารเช้าที่ดีและบริการส่วนตัวที่มีคุณภาพสูง มันเป็นที่นิยมมากในขณะที่มันอยู่ใกล้กับใจกลางของเมือง ฉันจะแนะนําให้พิจารณาการเยี่ยมชมโรงแรมนี้เพื่อความสะดวกสบาย
User:  ok
BOT:  คุณสามารถบอกฉันเพิ่มเติมเกี่ยวกับสถานที่หรือไม่?
Roseys ที่อยู่ใกล้เคียงรวมถึงThe Peninsula Bangkok, The Regent Hotel Bangkok, The Royal Orchid Bangkok และ The Grand Palace Bangkok

-> Bot just understand that it should output eng word, but rosey is the only english word in dataset
```

```
# No Context is train
User:  แนะนำหนังให้หน่อย
BOT: 
1. The Godfather
2. The Goddess
3. The God of Small Things
4. The God's Own Country
5. The Godless
6. The Good Shepherd
7. The Great Gatsby
8. The Girl with the Dragon Tattoo
9. The Great Train Robbery
10. The Great Wall
11. The Great White North
User:  ขอใหม่ๆได้ไหม
BOT: ""
```


```
Overfit / because old knowledge is not enough
User:  สวัสดี
BOT: 
1. ฉันสามารถให้คําแนะนําเกี่ยวกับการใช้ทรัพยากรออนไลน์ได้หรือไม่
2. มีวิธีใดเพื่อปรับปรุงความสามารถในการเขียนโปรแกรมคอมพิวเตอร์ของฉันได้หรือคุณสามารถให้คะแนนความสามารถในการเข้ารหัสและการเข้ารูปแบบได้หรอ
3. คุณสามารถช่วยให้
```

```
Don't know word / not smooth spaces
User:  แนะนำหนังรักให้หน่อย
BOT: 
1. The Hunger Games: Catching Fire
2. The Lord of the Rings: The Return of the King
3. Harry Potter and the Philosopher's Stone
4. The Dark Knight Trilogy
5. The Hungover Mommy
6. The Great Gatsby
7. A Beautiful Mind
8. The Godfather Part II
9. The Secret Life of Bees
10. The Hungry Caterpillar
```

```
# Pretrained data is bad?, so that it cannot link anything
User:  เขียนจดหมายสมัครงานให้หน่อย
BOT: 

ฉันเสียใจ แต่ฉันไม่สามารถช่วยคุณกับข้อมูลใด ๆ เกี่ยวกับการสมัครงานของคุณ โปรดติดต่อหน่วยงานบังคับใช้กฎหมายของสหรัฐอเมริกาโดยตรงเพื่อขอความช่วยเหลือ
```

```
# It look like fine-tune data is override everything in the model? 
User:  สวัสดี
BOT: 
1. ฉันสามารถทําให้คุณมีความสุขได้อย่างไร
2. คุณสามารถบอกฉันว่าคุณต้องการพูดอะไร
3. ฉิน, คุณรู้สึกเหมือนคุณมาถึง
4. ไม่, ฉันไม่คิดว่าฉันจะกลัว
5. ฉะนั้นฉันต้องการที่จะทําลายชีวิ
```

```
Do prompting in thai, but answer in english will help cross-language knowledge sharing?
[User THAI <----> BOT (ENG)] -> Translate -> User
[] -> scope
Try see, translation paper / multi-language lm paper
xlm, m2m100, nllb-200, ...
```

```
ท้องฟ้าเป็นสีม่วงเพราะอะไร -> Not make any sense
ท้องฟ้าเป็นสีม่วงเพราะอะไร? -> Have some sense
```

```
mgpt generated more footer noise 
-> วิธีการรักษาความปลอดภัยทางสังคมของผู้เข้าชมเว็บไซต์ Copyright © 2019 Posted by admin in Uncategorized " Dropship 1 1001 2 (อัปเดตล่าสุด
-> ...
```

```
Do something like train reward model -> train generative -> sample generative -> train another reward model (with original data & new sample data & first generation data?)
-> next step (train model to predict what sample to be choose / or may be using rule-based / model-based NLU algorithm)
```


```
Domain that is good
-> domain that can mixing with english knowledge (Movie, Hotel?), because english knowledge is very good
```
