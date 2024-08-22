from typing import List
import argparse
import datasets
import numpy
import os

import logging

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


ZERO_SHOT = os.environ.get('HOA_ZERO_SHOT', "False") == "True"


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

test = {"Status": "OK", "output": [
    {"Self-direction: thought attained": 0.0024574564304202795, "Self-direction: thought constrained": 0.00011691349209286273, "Self-direction: action attained": 0.0007427241653203964, "Self-direction: action constrained": 0.0009987016674131155, "Stimulation attained": 0.0016534713795408607, "Stimulation constrained": 7.240273407660425e-05, "Hedonism attained": 0.00011946013546548784, "Hedonism constrained": 5.626973870676011e-05, "Achievement attained": 0.0007442916976287961, "Achievement constrained": 0.0002864353300537914, "Power: dominance attained": 0.00047256675316020846, "Power: dominance constrained": 0.0005503477295860648, "Power: resources attained": 0.0026424031239002943, "Power: resources constrained": 0.0005643037147819996, "Face attained": 0.0009248544811271131, "Face constrained": 0.000765619392041117, "Security: personal attained": 0.0007875649025663733, "Security: personal constrained": 0.00041763478657230735, "Security: societal attained": 0.0015754187479615211, "Security: societal constrained": 0.0006250546430237591, "Tradition attained": 0.00037790153874084353, "Tradition constrained": 0.0001758920552674681, "Conformity: rules attained": 0.0008725103107281029, "Conformity: rules constrained": 0.0006714258925057948, "Conformity: interpersonal attained": 0.0003606232057791203, "Conformity: interpersonal constrained": 0.0008310032426379621, "Humility attained": 0.000631237868219614, "Humility constrained": 0.5000576972961426, "Benevolence: caring attained": 0.00016925728414207697, "Benevolence: caring constrained": 0.0002605092595331371, "Benevolence: dependability attained": 0.0004898284678347409, "Benevolence: dependability constrained": 0.00010994300828315318, "Universalism: concern attained": 0.0008231507381424308, "Universalism: concern constrained": 0.00043963169446215034, "Universalism: nature attained": 0.012091834098100662, "Universalism: nature constrained": 0.0008385402616113424, "Universalism: tolerance attained": 0.0003907356585841626, "Universalism: tolerance constrained": 0.0006092724506743252},
    {"Self-direction: thought attained": 0.09426147490739822, "Self-direction: thought constrained": 0.0005233777337707579, "Self-direction: action attained": 0.003918548114597797, "Self-direction: action constrained": 0.00040401381556876004, "Stimulation attained": 0.026682263240218163, "Stimulation constrained": 0.0050332010723650455, "Hedonism attained": 0.0007003754726611078, "Hedonism constrained": 0.00013359410513658077, "Achievement attained": 0.01484023965895176, "Achievement constrained": 0.0032732528634369373, "Power: dominance attained": 0.0006786694866605103, "Power: dominance constrained": 0.0010993105825036764, "Power: resources attained": 0.002530301921069622, "Power: resources constrained": 0.0012299943482503295, "Face attained": 0.0010289846686646342, "Face constrained": 0.000711916305590421, "Security: personal attained": 0.0028435694985091686, "Security: personal constrained": 0.0011656589340418577, "Security: societal attained": 0.0017767588142305613, "Security: societal constrained": 0.0012105898931622505, "Tradition attained": 0.0007454279693774879, "Tradition constrained": 0.00047847192035987973, "Conformity: rules attained": 0.00042179939919151366, "Conformity: rules constrained": 0.000527711003087461, "Conformity: interpersonal attained": 0.0006553333369083703, "Conformity: interpersonal constrained": 0.0019901844207197428, "Humility attained": 0.0025867691729217768, "Humility constrained": 0.5001301169395447, "Benevolence: caring attained": 0.0007135719060897827, "Benevolence: caring constrained": 0.00018532299145590514, "Benevolence: dependability attained": 0.0022150827571749687, "Benevolence: dependability constrained": 0.00013899922487325966, "Universalism: concern attained": 0.003934443928301334, "Universalism: concern constrained": 0.00030335880001075566, "Universalism: nature attained": 0.02208559960126877, "Universalism: nature constrained": 0.0023718939628452063, "Universalism: tolerance attained": 0.0012924829497933388, "Universalism: tolerance constrained": 0.0005632561515085399}]}



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

    return check.tolist()


################################################################################
################################################################################

def predict_context_based(input_list: List) -> List:
    prediction_list = []

    prev_sent_1 = None
    prev_sent_2 = None

    for entry in tqdm(input_list):
        if entry.get('Sentence-ID', 1) == 1:
            prev_sent_1 = prev_sent_2 = None

        text = entry['Text']
        full_text = text
        if prev_sent_1 is None and prev_sent_2 is not None:
            full_text = prev_sent_2 + text
        elif prev_sent_1 is not None and prev_sent_2 is not None:
            full_text = prev_sent_1 + prev_sent_2 + text

        temp = {"Text": full_text, 'language': entry.get('language', lang_dict['EN'])}
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

    return prediction_list


def predict_zero_shot(input_list: List) -> List:
    data = datasets.Dataset.from_pandas(pd.DataFrame.from_records(input_list))\
        .map(preprocess_function, batched=True, load_from_cache_file=False).to(device)
    predictions, label, metrics = trainer.predict(data, metric_key_prefix="predict")
    confidence_values = thresholds_vectorized(predictions)

    return confidence_values


def predict(input_list: List, validate_input: bool = True) -> List:
    if len(input_list) == 0:
        return []
    if validate_input:
        data_list = []
        old_language = 0
        for i, entry in enumerate(input_list):
            if isinstance(entry, str):
                # default to english text
                current_language = lang_dict['EN']
                data_list.append({
                    'Sentence-ID': 1 if current_language != old_language else i+1,
                    'Text': entry,
                    'language': lang_dict['EN']
                })

            elif isinstance(entry, dict) and 'Text' in entry.keys() and isinstance(entry['Text'], str):

                if 'language' in entry.keys():
                    if isinstance(entry['language'], int) and 0 <= entry['language'] < len(lang_dict):
                        current_language = lang_dict[entry['language']]
                    elif isinstance(entry['language'], str):
                        current_language = lang_dict.get(entry['language'], None)
                    else:
                        current_language = None
                else:
                    current_language = lang_dict['EN']

                if current_language is None:
                    logging.error(f"Unrecognized language: {entry['language']}")
                    continue

                if current_language != old_language:
                    sentence_id = 1
                elif 'Sentence-ID' in entry.keys():
                    if isinstance(entry['Sentence-ID'], int) and 0 <= entry['Sentence-ID']:
                        sentence_id = entry['Sentence-ID']
                    else:
                        logging.error(f"Unrecognized Sentence-ID: {entry['Sentence-ID']}")
                        sentence_id = i+1
                else:
                    sentence_id = i+1

                data_list.append({
                    'Sentence-ID': sentence_id,
                    'Text': entry['Text'],
                    'language': current_language
                })
            else:
                logging.error(f'Unable to format input: {entry}')
    else:
        data_list = input_list

    if ZERO_SHOT:
        prediction_list = predict_zero_shot(data_list)
    else:
        prediction_list = predict_context_based(data_list)
    result = pd.DataFrame(prediction_list, columns=labels)
    return result.to_dict('records')


if __name__ == "__main__" or is_running_as_inference_server():
    finetuned_model = '/models/SotirisLegkas/multi-head-xlm-xl-tokens-38'
    finetuned_tokenizer = '/tokenizer/SotirisLegkas/multi-head-xlm-xl-tokens-38'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(finetuned_tokenizer)
    model = MultiHead_MultiLabel_XL.from_pretrained(finetuned_model, problem_type="multi_label_classification").to(device)
    trainer = transformers.Trainer(model=model, args=TrainingArguments(finetuned_model, disable_tqdm=not ZERO_SHOT))
    logging.info(f'Running {"zero-shot" if ZERO_SHOT else "context-based"} prediction')
    
    if not is_running_as_inference_server():

        input_directory, output_dir = get_input_directory_and_output_directory('./dataset', default_output='./output')
        output_file = os.path.join(output_dir, "run.tsv")

        prediction_dataset = load_dataset(input_directory)

        result = predict(prediction_dataset.to_dict('records'), validate_input=False)

        df_prediction = pandas.DataFrame(
            {'Text-ID': prediction_dataset['Text-ID'], 'Sentence-ID': prediction_dataset['Sentence-ID']})
        df_prediction = pd.concat([df_prediction, pd.DataFrame.from_records(result)], axis=1)

        write_tsv_dataframe(output_file, df_prediction)
