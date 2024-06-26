import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from transformers import RobertaModel, RobertaTokenizer
from transformers.modeling_outputs import TokenClassifierOutput,SequenceClassifierOutput
from torch.utils.data import DataLoader, Dataset
import datasets
import numpy
import os
import pandas
import sys
import tempfile
import torch
import transformers
import wandb
import argparse
from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction
from tqdm.auto import tqdm
from typing import Optional, Tuple, Union
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from transformers.modeling_outputs import TokenClassifierOutput,SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead
from custom_models.multi_head import MultiHead_MultiLabel,MultiHead_MultiLabel_XL, MultiHead_MultiLabel_DebertaV2

wandb.init(
    # set the wandb project where this run will be logged
    project='touche',name='test')

#################################################################################################################################################################
#################################################################################################################################################################

def preprocess_function(examples):
        batch = tokenizer(examples["Text"], padding='max_length', max_length=512, truncation=True,)
        return batch

def load_dataset(directory, tokenizer, load_labels=True):
    sentences_file_path = os.path.join(directory, "sentences.tsv")
    labels_file_path = os.path.join(directory, "labels.tsv")
    data_frame = pandas.read_csv(sentences_file_path, encoding="utf-8", sep="\t", header=0).dropna()
    data_frame = datasets.Dataset.from_dict(data_frame)
    encoded_sentences = data_frame.map( preprocess_function, batched=True, load_from_cache_file=False)
    data_frame = pandas.DataFrame({'Text-ID':data_frame['Text-ID'],'Sentence-ID':data_frame['Sentence-ID'],'Text':data_frame['Text']})
    if load_labels and os.path.isfile(labels_file_path):
        labels_frame = pandas.read_csv(labels_file_path, encoding="utf-8", sep="\t", header=0)
        labels_frame = pandas.merge(data_frame, labels_frame, on=["Text-ID", "Sentence-ID"])
        ######################################   FOR TASK 1  #########################################
        # merged_df = pandas.DataFrame()
        # merged_df['Text-ID'] = labels_frame['Text-ID']
        # merged_df['Sentence-ID'] = labels_frame['Sentence-ID']
        # merged_df['Text'] = labels_frame['Text']
        # for col_name in labels_frame.columns:
        #     if col_name.endswith('attained'):
        #         prefix = col_name[:-9]
        #         constrained_col_name = f'{prefix} constrained'
        #         merged_df[prefix] = labels_frame[col_name] + labels_frame[constrained_col_name]
        ##############################################################################################
        labels_matrix = numpy.zeros((labels_frame.shape[0], len(labels)))
        for idx, label in enumerate(labels):
            if label in labels_frame.columns:
                labels_matrix[:, idx] = (labels_frame[label] >= 0.5).astype(int)
        encoded_sentences = encoded_sentences.add_column("labels", labels_matrix.tolist())
        # encoded_sentences["labels"] = labels_matrix.tolist()
    # encoded_sentences = datasets.Dataset.from_dict(encoded_sentences)
    return encoded_sentences, data_frame["Text-ID"].to_list(), data_frame["Sentence-ID"].to_list()

#################################################################################################################################################################
#################################################################################################################################################################
values = [ "Self-direction: thought", "Self-direction: action", "Stimulation",  "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance" ]
labels = sum([[value + " attained", value + " constrained"] for value in values], [])

id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)} 

pretrained_model = 'facebook/xlm-roberta-xl'
# pretrained_model = 'xlm-roberta-large'
# pretrained_model = 'microsoft/deberta-v2-xxlarge'

tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)
training_dataset, training_text_ids, training_sentence_ids = load_dataset('/home/vasters/touche24/touche-Sotiris/data/final_training', tokenizer)
validation_dataset, validation_text_ids, validation_sentence_ids = load_dataset('/home/vasters/touche24/touche-Sotiris/data/final_validation', tokenizer)

#################################################################################################################################################################
#################################################################################################################################################################
def context_data_2(data,sep=' </s> '):
    data=data.to_pandas()
    progress_bar = tqdm(range(len(data)))
    new_text=[]
    for i in range(len(data)):
        if data['Sentence-ID'][i]==2:
            one_indices = [j for j, label in enumerate(data['labels'][i-1]) if label == 1]
            if one_indices:
                text = data['Text'][i-1]
                for k in one_indices:
                    text=text+f' <{id2label[k]}>'
                text = text+sep+data['Text'][i]
            else:
                text = data['Text'][i-1] + ' <NONE>'+ sep+data['Text'][i]
            new_text.append(text)
        elif data['Sentence-ID'][i]>2:
            one_indices = [j for j, label in enumerate(data['labels'][i-1]) if label == 1]
            two_indices = [j for j, label in enumerate(data['labels'][i-2]) if label == 1]
            if two_indices:
                text = data['Text'][i-2]
                for k in two_indices:
                    text=text+f' <{id2label[k]}>'
            else:
                text = data['Text'][i-2] + ' <NONE>'
            if one_indices:
                text=text+ sep+ data['Text'][i-1]
                for k in one_indices:
                    text=text+f' <{id2label[k]}>'
                text=text+sep+data['Text'][i]
            else:
                text = text +sep+data['Text'][i-1]+' <NONE>'+sep+data['Text'][i]
            new_text.append(text)
        else:
            new_text.append(data['Text'][i])
        progress_bar.update(1)
    data = data[['Text-ID', 'Sentence-ID', 'Text','labels']]
    data['Text']=new_text
    data=datasets.Dataset.from_pandas(data)
    data = data.map(preprocess_function, batched=True, load_from_cache_file=False)
    return data


def context_data_2_without_tokens(data):
    data=data.to_pandas()
    progress_bar = tqdm(range(len(data)))
    new_text=[]
    for i in range(len(data)):
        if data['Sentence-ID'][i]==2:
            text = data['Text'][i-1] +' </s> '+data['Text'][i]
            new_text.append(text)
        elif data['Sentence-ID'][i]>2:
            text = data['Text'][i-2] +' </s> '+data['Text'][i-1]+' </s> '+data['Text'][i]
            new_text.append(text)
        else:
            new_text.append(data['Text'][i])
        progress_bar.update(1)
    data = data[['Text-ID', 'Sentence-ID', 'Text','labels']]
    data['Text']=new_text
    data=datasets.Dataset.from_pandas(data)
    data = data.map(preprocess_function, batched=True, load_from_cache_file=False)
    return data

print('START CONTEXT TWO')
print(training_dataset)
print(validation_dataset)


additional_special_tokens = ["<NONE>"]+['<'+label+'>' for label in labels]    
tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
   
print('START TRAIN DATA')
training_dataset=context_data_2(training_dataset)

print('START VALIDATION DATA')
validation_dataset=context_data_2(validation_dataset)

#################################################################################################################################################################
#################################################################################################################################################################
def multi_label_metrics(predictions, labels, thresholds=[0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_true = labels
    metrics = {}
    best_scores = {str(label_idx): 0.0 for label_idx in range(labels.shape[1])}
    best_threshold = {'threshold_'+str(label_idx): 0.0 for label_idx in range(labels.shape[1])}

    max_f1=0
    for threshold in thresholds:
        # threshold_metrics = {}

        y_pred = numpy.zeros(probs.shape)
        y_pred[numpy.where(probs >= threshold)] = 1
        f1=f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        metrics[f'f1_macro_{threshold}'] = f1
        if f1>max_f1:
            max_f1=f1

        for label_idx in range(labels.shape[1]):
            y_pred = (probs[:, label_idx] >= threshold).to(torch.int)
            f1 = f1_score(y_true[:, label_idx], y_pred,average='binary')
            if f1>best_scores[str(label_idx)]:
                best_threshold['threshold_'+str(label_idx)]=threshold
            best_scores[str(label_idx)] = max(best_scores[str(label_idx)], f1)
        #     threshold_metrics[f'label_{label_idx}_f1_{threshold}'] = f1
        # metrics.update(threshold_metrics)
    metrics.update(best_threshold)
    metrics.update(best_scores)
    mean_f1 = numpy.mean(list(best_scores.values()))
    metrics['max_f1']=max_f1
    metrics['mean_f1'] = mean_f1
    return metrics

# def multi_label_metrics(predictions, labels, threshold=0.5):
#     # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
#     sigmoid = torch.nn.Sigmoid()
#     probs = sigmoid(torch.Tensor(predictions))
#     y_true = labels
#     metrics = {}
#     # next, use threshold to turn them into integer predictions
#     max_f1=0
#     for threshold in [0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
#         y_pred = numpy.zeros(probs.shape)
#         y_pred[numpy.where(probs >= threshold)] = 1
#         f1=f1_score(y_true=y_true, y_pred=y_pred, average='macro')
#         metrics[f'f1_macro_{threshold}'] = f1
#         if f1>max_f1:
#             max_f1=f1
#     metrics['max_f1']=max_f1
#     return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

#################################################################################################################################################################
#################################################################################################################################################################
training_dataset=training_dataset.to_pandas()
validation_dataset=validation_dataset.to_pandas()

lang_dict={ 'EN': 0,
            'EL': 1,
            'DE': 2,
            'TR': 3,
            'FR': 4,
            'BG': 5,
            'HE': 6,
            'IT': 7,
            'NL': 8 }
id2lang_dict={v: k for k, v in lang_dict.items()}

training_dataset['language'] = training_dataset['Text-ID'].apply(lambda x: lang_dict[x[:2]])
validation_dataset['language'] = validation_dataset['Text-ID'].apply(lambda x: lang_dict[x[:2]])

training_dataset=datasets.Dataset.from_pandas(training_dataset)
# training_dataset=training_dataset.shuffle(seed=2024)
validation_dataset=datasets.Dataset.from_pandas(validation_dataset)

# training_dataset=datasets.concatenate_datasets([training_dataset,validation_dataset])

#################################################################################################################################################################
#################################################################################################################################################################

train_args = transformers.TrainingArguments(
   output_dir='test-dir',
   evaluation_strategy= 'epoch',
   save_strategy= 'epoch',
   save_total_limit = 2, 
   learning_rate=5e-6,
   num_train_epochs=20,
   push_to_hub=False,
   metric_for_best_model = 'mean_f1',
   greater_is_better=True, 
   weight_decay=0.01,
   load_best_model_at_end=True,
   per_device_train_batch_size=4,
   per_device_eval_batch_size=4,
   overwrite_output_dir=True,
   bf16=True,
   bf16_full_eval=True,
   lr_scheduler_type='linear',
   warmup_ratio=0.01,
   seed=2024,
#    fsdp='full_shard'
   )

training_dataset=training_dataset.remove_columns(['Text-ID', 'Sentence-ID', 'Text'])
validation_dataset=validation_dataset.remove_columns(['Text-ID', 'Sentence-ID', 'Text'])


model = MultiHead_MultiLabel_XL.from_pretrained(pretrained_model, problem_type="multi_label_classification",num_labels=len(labels), id2label=id2label, label2id=label2id)
model.resize_token_embeddings(len(tokenizer))

trainer = transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        #data_collator=data_collator,
        callbacks=[transformers.EarlyStoppingCallback(5)],
    )

train_result = trainer.train(resume_from_checkpoint=None)
metrics = train_result.metrics
metrics["train_samples"] = len(training_dataset)
trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

evaluation = trainer.evaluate(eval_dataset=validation_dataset)
max_eval_samples =  len(validation_dataset)
evaluation["eval_samples"] = max_eval_samples
trainer.log_metrics("eval", evaluation)
trainer.save_metrics("eval", evaluation)