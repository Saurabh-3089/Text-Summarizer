#from nltk.translate.lepor import length_penalty
import torch.cuda

from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_path))
        self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer,
                             device=device)
        self.gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

    def predict(self, text: str):
        print("Get Text: ")
        print(text)

        output = self.pipe(text, truncation=True, **self.gen_kwargs)[0]["summary_text"]

        print("\nModel Summary:")
        print(output)

        return output