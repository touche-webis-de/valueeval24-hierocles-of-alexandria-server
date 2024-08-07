import argparse
import datasets
import numpy
import os
import pandas
import torch
import transformers
import pandas as pd
import csv
import traceback
from tqdm.auto import tqdm
import torch.utils.checkpoint
from custom_models.multi_head import MultiHead_MultiLabel_XL
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


################################################################################
################################################################################
def preprocess_function(examples):
    batch = tokenizer(examples["Text"], padding='max_length', max_length=512, truncation=True, )
    return batch


def load_dataset(directory, tokenizer, load_labels=True):
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    data_frame = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0).dropna()
    data_frame = datasets.Dataset.from_dict(data_frame)
    encoded_sentences = data_frame.map(preprocess_function, batched=True, load_from_cache_file=False)
    return encoded_sentences


def write_tsv_dataframe(filepath, dataframe):
    try:
        dataframe.to_csv(filepath, encoding='utf-8', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)
    except IOError:
        traceback.print_exc()


################################################################################
#########################################################   CONSTANTS   ########
################################################################################
values = ["Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
          "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition",
          "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring",
          "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance"]
labels = sum([[value + " attained", value + " constrained"] for value in values], [])

id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

lang_dict = {'EN': 0,
             'EL': 1,
             'DE': 2,
             'TR': 3,
             'FR': 4,
             'BG': 5,
             'HE': 6,
             'IT': 7,
             'NL': 8}
id2lang_dict = {v: k for k, v in lang_dict.items()}


################################################################################
################################################################################

def get_previous_labeled_sentence_if_exists(data, idx, offset, sentence_separator):
    text = ""
    if data['Sentence-ID'][idx] > offset:
        previous_idx = idx - offset
        text = data['Text'][previous_idx]

        label_indices = [j for j, label in enumerate(data['pred_labels'][previous_idx]) if label == 1]
        if label_indices:
            for k in label_indices:
                text = text + f' <{id2label[k]}>'
        else:
            text = text + ' <NONE>'

        text = text + sentence_separator
    return text

def context_data_2(data, idx, sep=' </s> '):
    if idx == 0:
      return data.select(range(1))

    data = data.to_pandas()
    new_text = get_previous_labeled_sentence_if_exists(data, idx, 2, sep)
    new_text = new_text + get_previous_labeled_sentence_if_exists(data, idx, 1, sep)
    new_text = new_text + data['Text'][idx]

    data = data[['Text-ID', 'Sentence-ID', 'Text', 'language', 'pred_labels']][idx:idx + 1]
    data['Text'] = [ new_text ]
    data = datasets.Dataset.from_pandas(data)
    data = data.map(preprocess_function, batched=True, load_from_cache_file=False)
    return data


def thresholds(data):
    data = torch.sigmoid(torch.tensor(data))
    check = torch.zeros_like(data)
    check[:, 0] = (data[:, 0] >= 0.1).float()
    check[:, 1] = (data[:, 1] >= 0.3).float()
    check[:, 2] = (data[:, 2] >= 0.25).float()
    check[:, 3] = (data[:, 3] >= 0.25).float()
    check[:, 4] = (data[:, 4] >= 0.25).float()
    check[:, 5] = (data[:, 5] >= 0.25).float()
    check[:, 6] = (data[:, 6] >= 0.35).float()
    check[:, 7] = (data[:, 7] >= 0.3).float()
    check[:, 8] = (data[:, 8] >= 0.35).float()
    check[:, 9] = (data[:, 9] >= 0.25).float()
    check[:, 10] = (data[:, 10] >= 0.35).float()
    check[:, 11] = (data[:, 11] >= 0.15).float()
    check[:, 12] = (data[:, 12] >= 0.2).float()
    check[:, 13] = (data[:, 13] >= 0.25).float()
    check[:, 14] = (data[:, 14] >= 0.1).float()
    check[:, 15] = (data[:, 15] >= 0.2).float()
    check[:, 16] = (data[:, 16] >= 0.1).float()
    check[:, 17] = (data[:, 17] >= 0.15).float()
    check[:, 18] = (data[:, 18] >= 0.2).float()
    check[:, 19] = (data[:, 19] >= 0.25).float()
    check[:, 20] = (data[:, 20] >= 0.3).float()
    check[:, 21] = (data[:, 21] >= 0.25).float()
    check[:, 22] = (data[:, 22] >= 0.3).float()
    check[:, 23] = (data[:, 23] >= 0.15).float()
    check[:, 24] = (data[:, 24] >= 0.1).float()
    check[:, 25] = (data[:, 25] >= 0.15).float()
    check[:, 26] = (data[:, 26] >= 0.1).float()
    check[:, 27] = (data[:, 27] >= 0.0).float()
    check[:, 28] = (data[:, 28] >= 0.3).float()
    check[:, 29] = (data[:, 29] >= 0.1).float()
    check[:, 30] = (data[:, 30] >= 0.15).float()
    check[:, 31] = (data[:, 31] >= 0.4).float()
    check[:, 32] = (data[:, 32] >= 0.15).float()
    check[:, 33] = (data[:, 33] >= 0.2).float()
    check[:, 34] = (data[:, 34] >= 0.1).float()
    check[:, 35] = (data[:, 35] >= 0.1).float()
    check[:, 36] = (data[:, 36] >= 0.25).float()
    check[:, 37] = (data[:, 37] >= 0.2).float()
    return check


################################################################################
################################################################################


def parse_args():
    parser = argparse.ArgumentParser("predict.py")

    parser.add_argument('-s', '--sentences-dir', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    finetuned_model = '/models/SotirisLegkas/multi-head-xlm-xl-tokens-38'
    finetuned_tokenizer = '/tokenizer/SotirisLegkas/multi-head-xlm-xl-tokens-38'
    tokenizer = transformers.AutoTokenizer.from_pretrained(finetuned_tokenizer)
    model = MultiHead_MultiLabel_XL.from_pretrained(finetuned_model, problem_type="multi_label_classification")
    trainer = transformers.Trainer(model=model)

    validation_dataset = load_dataset(
        args.sentences_dir, tokenizer, load_labels=False)

    validation_dataset = validation_dataset.to_pandas()

    validation_dataset['language'] = validation_dataset['Text-ID'].apply(lambda x: lang_dict[x[:2]])
    validation_dataset['pred_labels'] = None
    validation_dataset = datasets.Dataset.from_pandas(validation_dataset)


    for i in tqdm(range(0, len(validation_dataset))):
        temp = context_data_2(validation_dataset, i)
        predictions, label, metrics = trainer.predict(temp, metric_key_prefix="predict")
        check = thresholds(predictions)
        validation_dataset = validation_dataset.to_pandas()
        validation_dataset['pred_labels'][i] = check.tolist()[0]
        validation_dataset = datasets.Dataset.from_pandas(validation_dataset)

    result = pd.DataFrame(validation_dataset['pred_labels'], columns=labels)

    df_prediction = pandas.DataFrame(
        {'Text-ID': validation_dataset['Text-ID'], 'Sentence-ID': validation_dataset['Sentence-ID']})
    df_prediction = pd.concat([df_prediction, result], axis=1)

    write_tsv_dataframe(args.output_file, df_prediction)

