import os.path
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch

from textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params: ModelTrainerConfig):
        self.config = config
        self.params = params

    def train(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir= str(self.config.root_dir), num_train_epochs= self.params.num_train_epochs,
            warmup_steps= self.params.warmup_steps,
            per_device_train_batch_size= self.params.per_device_train_batch_size,
            weight_decay= self.params.weight_decay, logging_steps= self.params.logging_steps,
            eval_strategy= self.params.evaluation_strategy, eval_steps= self.params.eval_steps,
            save_steps= self.params.save_steps,
            gradient_accumulation_steps= self.params.gradient_accumulation_steps
        )

        trainer = Trainer(model=model_pegasus, args=trainer_args, tokenizer=tokenizer,
                          data_collator=seq2seq_data_collator, train_dataset=dataset_samsum_pt['test'],
                          eval_dataset=dataset_samsum_pt['validation'])

        trainer.train()

        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))

        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

