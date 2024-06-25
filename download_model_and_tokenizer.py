from os import path
from custom_models.multi_head import MultiHead_MultiLabel_XL
from transformers import (AutoConfig, AutoTokenizer)


finetuned_model ='SotirisLegkas/multi-head-xlm-xl-tokens-38'


def download_tokenizer(tokenizer_dir: str = "/tokenizer"):
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model)
    config = AutoConfig.from_pretrained(finetuned_model)
    
    tokenizer.save_pretrained(path.join(tokenizer_dir, finetuned_model))
    config.save_pretrained(path.join(tokenizer_dir, finetuned_model))


def download_model(models_dir: str = "/models"):
    model = MultiHead_MultiLabel_XL.from_pretrained(finetuned_model, problem_type="multi_label_classification")

    model.save_pretrained(path.join(models_dir, finetuned_model))


if __name__ == "__main__":
    download_model()
    download_tokenizer()

