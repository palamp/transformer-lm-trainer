from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, default_data_collator, set_seed
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from dataset.dataset import EOSSplitTextDataset, CloneDataset
from omegaconf import OmegaConf
import os
from typing import List
import shutil
from utils_v2 import get_result_dir, on_after_train, is_main_process
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
set_seed(42)


class GenerationCallback(TrainerCallback):

    def __init__(self, config, result_dir: str) -> None:
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        self.generate_config = self.config.get('generate', {})
        self.log_example: List[str] = self.generate_config.get('examples', [])

        print('Will visualize this on going')
        print(self.log_example)

    def _generate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: str) -> str:
        inputs = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(inputs.to(model.device),
                                 no_repeat_ngram_size=4,
                                 do_sample=True,
                                 top_p=0.95,
                                 temperature=0.5,
                                 top_k=4,
                                 repetition_penalty=1.03,
                                 penalty_alpha=0.6,
                                 eos_token_id=tokenizer.eos_token_id,
                                 **self.generate_config.get('config', {})
                                 )
        decode_text = tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        return decode_text

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, logs=None, **kwargs):
        if state.is_local_process_zero:
            os.makedirs(self.result_dir, exist_ok=True)
            with open(f'{self.result_dir}/examples.txt', 'a') as w:
                w.write(
                    f'=================={state.global_step}==================\n')
                for example in self.log_example:
                    generated_text = self._generate(model, tokenizer, example)
                    remove_newline_g_text = ' '.join(
                        generated_text.split('\n'))
                    w.write(f'{remove_newline_g_text}\n')


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
        generation_callback = GenerationCallback(self.config, result_dir)
        if is_main_process():
            total_train_params = sum(p.numel()
                                     for p in self.model.parameters() if p.requires_grad)
            print(f'Total train param: {total_train_params:,}')
        self.eval_steps = self.config.training_args.pop('eval_steps')
        self.train_dataset = self._get_train_dataset()
        self.eval_dataset = self._get_eval_dataset()
        training_args = TrainingArguments(
            do_train=True, do_eval=True, evaluation_strategy='steps', output_dir=result_dir, dataloader_num_workers=0, learning_rate=self.config.lr, eval_steps=self.eval_steps, **self.config.training_args)
        super().__init__(self.model, training_args, default_data_collator, self.train_dataset, self.eval_dataset,
                         self.tokenizer, callbacks=[generation_callback], **config.get('trainer_args', {}))

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
        test_path = data_config.get('test_path', None)
        if test_path is None:
            eval_dataset = CloneDataset(self.train_dataset)
            self.eval_steps = 10
            return eval_dataset
        eval_dataset = EOSSplitTextDataset(test_path, tokenizer_name=self.config.get(
            'tokenizer_name', self.config.model_name), arch='prefix_lm' if self.is_enc_dec else 'clm', **data_config.get('config', {}))
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
