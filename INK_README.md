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
TH->EN answer
#### Very good, accurate answer
Good accurate answer, however may need to repeat the question
User:  แนะนำโรงแรมในกรุงเทพให้หน่อย
BOT:  1. The Savoy Hotel Bangkok is a great choice for an elegant stay in the city. 2. Another popular option would be the Renaissance Grand Central Tokyo, which offers modern luxury with stylish decor and excellent service at reasonable prices. 3. If budget isn't your priority but you want to experience some of London’s most iconic landmarks like Buckingham Palace or Tower Bridge then check out the Four Seasons Hotel London St Pancras Airport – it has all that plus free Wi-Fi! 4. For something more affordable yet luxurious try the Mandarin Oriental, offering contemporary design combined with thoughtful amenities such as gourmet dining options
User:  ขอในกรุงเทพ
BOT: The Shangri La Hotel, located on Sukhumvit Soi 25, is one of my top recommendations if looking into staying near downtown Bangkok while exploring its vibrant nightlife district. It also features complimentary WiFi throughout each guestroom so no matter what time zone they are in there won't be any lag when accessing their favorite websites online.
User:  ราคาเท่าไหร่
BOT: Shangrila Hotel Bangkok rates start from around THB 2,500 per room/night depending upon availability during peak season (December - April).
```


```
Domain that is good
-> domain that can mixing with english knowledge (Movie, Hotel?), because english knowledge is very good
```


### Pretraining
```
https://arxiv.org/pdf/2112.00861.pdf (page=28)
55% heavily
filtered common crawl data (220B tokens),
32% internet books (128B tokens)
OpenWebText, Wikipedia, Stack Exchange, Arxiv, Legal and Patent documents, Ubuntu-IRC
discussion, and movie scripts, most of which we sourced from The Pile [GBB+20].

finetuned for 100B tokens on a distribution of python code containing about
45B unique tokens,

AALM layer=64 d=8192 52B flops=1.2e23 <---> GPT-3 flops=3.14e23

They PMP on page=(20, 34)
They using PM loss + LM loss & train on binary dataset?
Which can reduce number of fine-tuning pairs into 100 <--> 1,000 (@ 10k with pretrained / without is similar) (difference by difficulty of tasks), but most of them are "ok" at 10k datapoint
StackExchange: The StackExchange Data Dump17 -> 5.8M training pairs
REDDIT: Pushshift reddit -> 1.1M training pairs
Wiki:  -> 1.4 training pairs
Mix -> 70% StackExchange, ...
batch-size=32, lr = 0.01 * pretrained_lr

```


### Pretrained target / cost
```
def calc_loss(n, d):
    loss = 1.69 + (406.4 / (pow(n, 0.34))) + (410.7 / (pow(d, 0.27)))
    return loss
# ~ optimal point for 20B model
# n = 23 * 1_000_000_000
# d = 550 * 1_000_000_000
# calc_loss(n, d)
# loss
# ~ (20 token per params optimal) ratio
# gpt-3 ~ 2.078
# anthropiclm (52B * 400B) ~ 2.085
# baseline (7B model) xglm ~ 2.160363238612205
# 27B * 540B ~ 2.084 / 26B * 520B ~ 2.088
# 10B * 205B ~ 2.21
# gpt-neox -> 20B * 472B ~ 2.10
# roberta is train on 400B tokens
# 13B * 260B ~ 2.178 (3_000_000 baht)
# 6.7B * 134B ~ 2.28 (worse than xglm)
# 20B * 400B (optimal for 20B) (gpt-neox should be better)
# but 12B is almost same as 20B ~?
# 12B * 300B (pythia) ~ 2.169
# small model
# 1.4B * 300B (pythia) ~ 2.33
# 2.7B * 54B ~ 2.46 (180_000 baht) (much more than pythia)
# 6.7B * 134B ~ 2.28 (900_000 baht) (little better than pythia)
# or 2.7B * 134B ~ 2.34 (? 450_000 baht) (same as pythia)
# 66B OPT * 300B ~ 2.10 (same as gpt-neox)
# 66B * 400B ~ 2.078 = gpt-3
```