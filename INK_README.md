```
4xA100_80G
> 16bs @ 280 max_length
> 8bs @ 1024 max_length
4bs @ 2048 max_length
# Model checkpoint
7.5B model need hdd=126 per checkpoint (98) for checkpoint, 28 for weight
```


```
# google deeplearning vm suck
rm -rf ~/.cache/torch_extensions/
conda create --name myenv python=3.9
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


BS=32 * accum=2 [02:55<13:22:40,  9.61s/it] (use ~47 GB of gpu ram & 120GB of ram)
$3.93 per GPU
```

```
# GCP VM NVME https://cloud.google.com/compute/docs/disks/add-local-ssd
lsblk
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/disks/nvme
sudo mount /dev/nvme0n1 /mnt/disks/nvme
sudo chmod a+w /mnt/disks/nvme/
sudo vim /etc/fstab

echo UUID=`sudo blkid -s UUID -o value /dev/disk/by-id/google-local-nvme-ssd-0` /mnt/disks/nvme ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
UUID=1c2c1d50-61e3-4fc3-87f8-d14f1299f8e0 /mnt/disks/nvme ext4 discard,defaults,nofail 0 2
```

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

```
# with no pad-change to eos, with pad calc for loss
[2023-02-08 13:52:09,860] [INFO] [config.py:997:print_user_config]   json = {
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-05,
            "betas": [0.9, 0.95],
            "eps": 1e-08,
            "weight_decay": 0.0
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-05,
            "warmup_num_steps": 20
        }
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2.000000e+08,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2.000000e+08,
        "contiguous_gradients": true,
        "cpu_offload": true
    },
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 0.5,
    "steps_per_print": 2.000000e+03,
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 32,
    "wall_clock_breakdown": false
}
Using /home/kunato/.cache/torch_extensions/py39_cu113 as PyTorch extensions root...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Time to load utils op: 0.0004189014434814453 seconds
***** Running training *****
  Num examples = 160929
  Num Epochs = 1
  Instantaneous batch size per device = 32
  Total train batch size (w. parallel, distributed & accumulation) = 64
  Gradient Accumulation steps = 2
  Total optimization steps = 2515
  Number of trainable parameters = 7492771840
  0%|                                                                                                                                                                   | 0/2515 [00:00<?, ?it/s][2023-02-08 13:52:24,187] [INFO] [stage_1_and_2.py:1762:step] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 65536, reducing to 65536
  0%|                                                                                                                                                        | 1/2515 [00:14<10:00:14, 14.33s/it][2023-02-08 13:52:33,452] [INFO] [stage_1_and_2.py:1762:step] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 65536, reducing to 32768.0
  0%|                                                                                                                                                         | 2/2515 [00:23<7:55:18, 11.35s/it][2023-02-08 13:52:42,731] [INFO] [stage_1_and_2.py:1762:step] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 32768.0, reducing to 16384.0
  0%|▏                                                                                                                                                        | 3/2515 [00:32<7:15:34, 10.40s/it][2023-02-08 13:52:52,029] [INFO] [stage_1_and_2.py:1762:step] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16384.0, reducing to 8192.0
  0%|▏                                                                                                                                                        | 4/2515 [00:42<6:57:06,  9.97s/it][2023-02-08 13:53:01,315] [INFO] [stage_1_and_2.py:1762:step] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
  0%|▎                                                                                                                                                        | 5/2515 [00:51<6:46:40,  9.72s/it][2023-02-08 13:53:10,598] [INFO] [stage_1_and_2.py:1762:step] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4096.0, reducing to 2048.0
{'loss': 10.6234, 'learning_rate': 4.6275642631951835e-06, 'epoch': 0.0}
{'loss': 8.6182, 'learning_rate': 8.809389787307027e-06, 'epoch': 0.01}
{'loss': 7.3904, 'learning_rate': 1e-05, 'epoch': 0.01}
{'loss': 6.5842, 'learning_rate': 1e-05, 'epoch': 0.02}
{'loss': 6.4482, 'learning_rate': 1e-05, 'epoch': 0.02}
{'loss': 6.3746, 'learning_rate': 1e-05, 'epoch': 0.02}
{'loss': 5.9596, 'learning_rate': 1e-05, 'epoch': 0.03}
{'loss': 5.6346, 'learning_rate': 1e-05, 'epoch': 0.03}
{'loss': 5.4766, 'learning_rate': 1e-05, 'epoch': 0.04}
{'loss': 5.1166, 'learning_rate': 1e-05, 'epoch': 0.04}
{'loss': 4.8258, 'learning_rate': 1e-05, 'epoch': 0.04}
{'loss': 4.5961, 'learning_rate': 1e-05, 'epoch': 0.05}
{'loss': 4.2157, 'learning_rate': 1e-05, 'epoch': 0.05}
{'loss': 3.8688, 'learning_rate': 1e-05, 'epoch': 0.06}
{'loss': 3.6674, 'learning_rate': 1e-05, 'epoch': 0.06}
{'loss': 3.3288, 'learning_rate': 1e-05, 'epoch': 0.06}
{'loss': 3.1702, 'learning_rate': 1e-05, 'epoch': 0.07}
{'loss': 3.2118, 'learning_rate': 1e-05, 'epoch': 0.07}
{'loss': 3.091, 'learning_rate': 1e-05, 'epoch': 0.08}
{'loss': 3.1509, 'learning_rate': 1e-05, 'epoch': 0.08}
  8%|████████████                                                                                                                                           | 200/2515 [46:16<9:01:34, 14.04s/it]***** Running Evaluation *****
  Num examples = 1624
  Batch size = 8
{'eval_loss': 0.7646484375, 'eval_runtime': 38.6124, 'eval_samples_per_second': 42.059, 'eval_steps_per_second': 5.257, 'epoch': 0.08}
{'loss': 2.9816, 'learning_rate': 1e-05, 'epoch': 0.08}
{'loss': 3.0971, 'learning_rate': 1e-05, 'epoch': 0.09}
{'loss': 3.2587, 'learning_rate': 1e-05, 'epoch': 0.09}
{'loss': 3.0206, 'learning_rate': 1e-05, 'epoch': 0.1}
{'loss': 3.452, 'learning_rate': 1e-05, 'epoch': 0.1}
{'loss': 3.0924, 'learning_rate': 1e-05, 'epoch': 0.1}
{'loss': 3.5004, 'learning_rate': 1e-05, 'epoch': 0.11}
{'loss': 3.4638, 'learning_rate': 1e-05, 'epoch': 0.11}
{'loss': 3.2218, 'learning_rate': 1e-05, 'epoch': 0.12}
{'loss': 3.5955, 'learning_rate': 1e-05, 'epoch': 0.12}
{'loss': 3.6354, 'learning_rate': 1e-05, 'epoch': 0.12}
{'loss': 3.4691, 'learning_rate': 1e-05, 'epoch': 0.13}
{'loss': 3.5324, 'learning_rate': 1e-05, 'epoch': 0.13}
{'loss': 3.6847, 'learning_rate': 1e-05, 'epoch': 0.14}
{'loss': 3.5771, 'learning_rate': 1e-05, 'epoch': 0.14}
{'loss': 3.7693, 'learning_rate': 1e-05, 'epoch': 0.14}
{'loss': 3.5253, 'learning_rate': 1e-05, 'epoch': 0.15}
{'loss': 3.5966, 'learning_rate': 1e-05, 'epoch': 0.15}
{'loss': 3.7167, 'learning_rate': 1e-05, 'epoch': 0.16}
{'loss': 3.8509, 'learning_rate': 1e-05, 'epoch': 0.16}
 16%|███████████████████████▋                                                                                                                             | 400/2515 [1:33:28<8:10:14, 13.91s/it]***** Running Evaluation *****
  Num examples = 1624
  Batch size = 8
{'eval_loss': 0.5908203125, 'eval_runtime': 38.5426, 'eval_samples_per_second': 42.135, 'eval_steps_per_second': 5.267, 'epoch': 0.16}
{'loss': 3.5652, 'learning_rate': 1e-05, 'epoch': 0.16}
{'loss': 3.6688, 'learning_rate': 1e-05, 'epoch': 0.17}
{'loss': 3.9716, 'learning_rate': 1e-05, 'epoch': 0.17}
{'loss': 3.9224, 'learning_rate': 1e-05, 'epoch': 0.17}
{'loss': 3.7126, 'learning_rate': 1e-05, 'epoch': 0.18}
{'loss': 3.9527, 'learning_rate': 1e-05, 'epoch': 0.18}
{'loss': 3.8443, 'learning_rate': 1e-05, 'epoch': 0.19}
{'loss': 3.8986, 'learning_rate': 1e-05, 'epoch': 0.19}
{'loss': 4.0804, 'learning_rate': 1e-05, 'epoch': 0.19}
{'loss': 4.0222, 'learning_rate': 1e-05, 'epoch': 0.2}
{'loss': 3.9105, 'learning_rate': 1e-05, 'epoch': 0.2}
{'loss': 4.0428, 'learning_rate': 1e-05, 'epoch': 0.21}
{'loss': 4.1126, 'learning_rate': 1e-05, 'epoch': 0.21}
{'loss': 4.1881, 'learning_rate': 1e-05, 'epoch': 0.21}
{'loss': 4.0692, 'learning_rate': 1e-05, 'epoch': 0.22}
{'loss': 4.0088, 'learning_rate': 1e-05, 'epoch': 0.22}
```