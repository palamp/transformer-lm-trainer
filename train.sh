# deepspeed --include="localhost:1" trainer.py --config config/config_clm_story.yaml
# deepspeed --num_gpus=2 trainer.py --config config/config_clm_story.yaml
# deepspeed --num_gpus=2 trainer.py --config config/config_encdec.yaml
# deepspeed --num_gpus=2 trainer.py --config config/config_clm_story.yaml
# python trainer.py
# deepspeed --include="localhost:0" trainer.py --config config/config_paraphase.yaml
CUDA_VISIBLE_DEVICES=1 python trainer.py --config config/config_paraphase_mt5.yaml