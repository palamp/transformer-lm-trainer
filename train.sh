# deepspeed --num_gpus=2 trainer.py --config config/config_clm.yaml
deepspeed --num_gpus=2 trainer.py --config config/config_encdec.yaml
# python trainer.py