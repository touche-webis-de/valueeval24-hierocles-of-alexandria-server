import argparse
import datasets
import numpy
import os

import numpy as np
import pandas
import torch
import transformers
from transformers.training_args import TrainingArguments
import pandas as pd
import csv
import traceback
from tqdm.auto import tqdm
import torch.utils.checkpoint
from custom_models.multi_head import MultiHead_MultiLabel_XL
from collections import Counter
from tira.third_party_integrations import is_running_as_inference_server, get_input_directory_and_output_directory
import warnings

warnings.filterwarnings("ignore")
datasets.utils.logging.disable_progress_bar()


ZERO_SHOT = bool(os.environ.get('HOA_ZERO_SHOT', False))


################################################################################
################################################################################

def tokenize(text: str):
    return tokenizer(text, padding='max_length', max_length=512, truncation=True)


def preprocess_function(examples):
    batch = tokenizer(examples["Text"], padding='max_length', max_length=512, truncation=True)
    return batch


def load_dataset(directory):
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    data_frame = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0).dropna()
    data_frame['language'] = data_frame['Text-ID'].apply(lambda x: lang_dict[x[:2]])
    return data_frame


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

label_thresholds = [
    0.10, 0.30, 0.25, 0.25, 0.25, 0.25, 0.35, 0.30, 0.35, 0.25,
    0.35, 0.15, 0.20, 0.25, 0.10, 0.20, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.25, 0.30, 0.15, 0.10, 0.15, 0.10, 0.00, 0.30, 0.10,
    0.15, 0.40, 0.15, 0.20, 0.10, 0.10, 0.25, 0.20
]

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

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def map_to_confidence(x, thresh):
    if -0.00000001 <= x <= 1.00000001:
        if x >= thresh:
            return (x - thresh) / (thresh - 1) * (-0.5) + 0.5
        else:
            return x / thresh * 0.5
    else:
        raise ValueError(f"{x} outside of interval [0,1].")


def thresholds(data):
    data = sigmoid(data)
    tuples = zip(data.tolist(), label_thresholds)
    confidence = [map_to_confidence(x, thresh) for x, thresh in tuples]

    return confidence


def thresholds_vectorized(data):
    data = torch.sigmoid(torch.tensor(data))
    check = torch.zeros_like(data)
    for i, thresh in enumerate(label_thresholds):
        check[:, i] = torch.from_numpy(np.vectorize(
            lambda x: map_to_confidence(x, thresh)
        )(data[:, i]))

    return check.tolist()[0]


################################################################################
################################################################################


if __name__ == "__main__" or is_running_as_inference_server():
    finetuned_model = '/models/SotirisLegkas/multi-head-xlm-xl-tokens-38'
    finetuned_tokenizer = '/tokenizer/SotirisLegkas/multi-head-xlm-xl-tokens-38'
    tokenizer = transformers.AutoTokenizer.from_pretrained(finetuned_tokenizer)
    model = MultiHead_MultiLabel_XL.from_pretrained(finetuned_model, problem_type="multi_label_classification")
    trainer = transformers.Trainer(model=model, args=TrainingArguments(finetuned_model, disable_tqdm=True))
    
    if not is_running_as_inference_server():

        input_directory, output_dir = get_input_directory_and_output_directory('./dataset', default_output='./output')
        output_file = os.path.join(output_dir, "run.tsv")

        prediction_dataset = load_dataset(input_directory)

        prediction_list = []

        prev_sent_1 = None
        prev_sent_2 = None

        for i in tqdm(range(len(prediction_dataset))):
            if prediction_dataset.at[i, 'Sentence-ID'] == 1:
                prev_sent_1 = prev_sent_2 = None

            text = prediction_dataset.at[i, "Text"]
            full_text = text
            if prev_sent_1 is None and prev_sent_2 is not None:
                full_text = prev_sent_2 + text
            elif prev_sent_1 is not None and prev_sent_2 is not None:
                full_text = prev_sent_1 + prev_sent_2 + text
            
            temp = {"Text": full_text, 'language': int(prediction_dataset["language"][i])}
            temp.update(tokenize(full_text))
            
            temp = datasets.Dataset.from_dict({key: [value] for key, value in temp.items()})
            
            predictions, label, metrics = trainer.predict(temp, metric_key_prefix="predict")
            # predictions is numpy.ndarray
            
            pred_values = predictions[0]
            confidence_values = thresholds(pred_values)

            prediction_list.append(confidence_values)
            
            prev_sent_1 = prev_sent_2
            
            text_labels = [f"<{id2label[j]}>" for j, value in enumerate(confidence_values) if value >= 0.5]

            if len(text_labels) > 0:
                prev_sent_2 = f"{text} <{' '.join(text_labels)}> </s> "
            else:
                prev_sent_2 = f"{text} <NONE> </s> "
            
        result = pd.DataFrame(prediction_list, columns=labels)

        df_prediction = pandas.DataFrame(
            {'Text-ID': prediction_dataset['Text-ID'], 'Sentence-ID': prediction_dataset['Sentence-ID']})
        df_prediction = pd.concat([df_prediction, result], axis=1)

        write_tsv_dataframe(output_file, df_prediction)

