from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, set_seed
from dataset.dataset import EOSSplitTextDataset
from omegaconf import OmegaConf
import os
import shutil
from utils_v2 import get_result_dir, on_after_train, is_main_process
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
set_seed(42)


class CustomTrainer(Trainer):
    def __init__(self, config, result_dir: str):
        self.config = config
        self.is_enc_dec = 't5' in self.config.model_name
        model_config = AutoConfig.from_pretrained(self.config.model_name)
        model_config.gradient_checkpointing = True
        model_config.use_cache = False

        if self.is_enc_dec:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name, config=model_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, config=model_config)
        self._set_tokenizer()
        self.model.resize_token_embeddings(len(self.tokenizer))

        if is_main_process():
            total_train_params = sum(p.numel()
                                     for p in self.model.parameters() if p.requires_grad)
            print(f'Total train param: {total_train_params:,}')
        training_args = TrainingArguments(
            do_train=True, do_eval=True, evaluation_strategy='steps', output_dir=result_dir, dataloader_num_workers=0, learning_rate=self.config.lr, **self.config.training_args)
        super().__init__(self.model, training_args, default_data_collator, self._get_train_dataset(),
                         self._get_eval_dataset(), self.tokenizer, **config.get('trainer_args', {}))

    def _set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('tokenizer_name', self.config.model_name))

    def _get_train_dataset(self):
        data_config = self.config.data
        train_dataset = EOSSplitTextDataset(
            data_config.train_path, tokenizer_name=self.config.get('tokenizer_name', self.config.model_name), arch='prefix_lm' if self.is_enc_dec else 'clm', **data_config.get('config', {}))
        return train_dataset

    def _get_eval_dataset(self):
        data_config = self.config.data
        eval_dataset = EOSSplitTextDataset(
            data_config.test_path, tokenizer_name=self.config.get('tokenizer_name', self.config.model_name), arch='prefix_lm' if self.is_enc_dec else 'clm', **data_config.get('config', {}))
        return eval_dataset


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    args = parser.parse_args()
    conf_file = args.config

    print(f'Config {conf_file}')
    conf = OmegaConf.load(conf_file)
    result_dir = get_result_dir()
    if is_main_process(args.local_rank):
        print('Creating: ', result_dir)
        os.makedirs(result_dir, exist_ok=False)
        shutil.copy(conf_file, f'{result_dir}/config.yaml')
    trainer = CustomTrainer(conf, result_dir)
    train_result = trainer.train()
    trainer.save_model()
    on_after_train(trainer, train_result)
