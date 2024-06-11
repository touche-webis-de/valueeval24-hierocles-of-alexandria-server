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
import torch
import datasets
import numpy
import os
import pandas
import torch
import transformers
from tqdm.auto import tqdm
import torch.utils.checkpoint
from custom_models.multi_head import MultiHead_MultiLabel, MultiHead_MultiLabel_XL, MultiHead_MultiLabel_DebertaV2
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
#################################################################################################################################################################
#################################################################################################################################################################
def compute_positive_weights(df, labels):
    counter = Counter()
    for label_list in df['labels']:
        for idx, label in enumerate(label_list):
            if label == 1:
                counter[labels[idx]] += 1
    total = len(df.index)
    pos_weights = []
    for label in labels:
        if label in counter:  # Check if the label exists in the counter
            pos_weights.append(total / counter[label])
        else:
            pos_weights.append(0)  # If the label does not exist in the counter, assign a weight of 0
    print("Class positive weights: ", pos_weights)
    return pos_weights

def preprocess_function(examples):
        batch = tokenizer( examples["Text"], padding='max_length', max_length=512, truncation=True,)
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

def write_tsv_dataframe(filepath, dataframe):
    try:
        dataframe.to_csv(filepath, encoding='utf-8', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)
    except IOError:
        traceback.print_exc()

#################################################################################################################################################################
#################################################################################################################################################################
values = [ "Self-direction: thought", "Self-direction: action", "Stimulation",  "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance" ]
labels = sum([[value + " attained", value + " constrained"] for value in values], [])

id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)} 
#################################################################################################################################################################
#################################################################################################################################################################
finetuned_model ='/home/vasters/touche24/touche-Sotiris/xlm-roberta-xl-38-layers-5e-6-all-dir/checkpoint-74580'  #"/home/sotirislegkas/touche/task-1_EL_1e-5_XLM-R-large_threshold_output_dir"
tokenizer = transformers.AutoTokenizer.from_pretrained(finetuned_model)

# validation_dataset, validation_text_ids, validation_sentence_ids = load_dataset('/home/vasters/touche24/touche-Sotiris/data/final_test-english', tokenizer)   #'/home/sotirislegkas/touche/data/validation'
validation_dataset, validation_text_ids, validation_sentence_ids = load_dataset('/home/vasters/touche24/touche-Sotiris/data/final_test', tokenizer)   #'/home/sotirislegkas/touche/data/validation'

#################################################################################################################################################################
#################################################################################################################################################################
validation_dataset=validation_dataset.to_pandas()
# training_dataset=training_dataset.to_pandas()
# weights=compute_positive_weights(training_dataset,labels)
model = MultiHead_MultiLabel_XL.from_pretrained(finetuned_model, problem_type="multi_label_classification")


model.push_to_hub('SotirisLegkas/multi-head-xlm-xl-tokens-38')


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
validation_dataset['language'] = validation_dataset['Text-ID'].apply(lambda x: lang_dict[x[:2]])
validation_dataset=datasets.Dataset.from_pandas(validation_dataset)
print(validation_dataset)
#################################################################################################################################################################
#################################################################################################################################################################
def context_data_2(data,idx,sep=' </s> '):
    data=data.to_pandas()
    new_text=[]
    if data['Sentence-ID'][idx]==2:
        one_indices = [j for j, label in enumerate(data['pred_labels'][idx-1]) if label == 1]
        if one_indices:
            text = data['Text'][idx-1]
            for k in one_indices:
                text=text+f' <{id2label[k]}>'
            text = text+sep+data['Text'][idx]
        else:
            text = data['Text'][idx-1] + ' <NONE>'+ sep+data['Text'][idx]
        new_text.append(text)
    elif data['Sentence-ID'][idx]>2:
        one_indices = [j for j, label in enumerate(data['pred_labels'][idx-1]) if label == 1]
        two_indices = [j for j, label in enumerate(data['pred_labels'][idx-2]) if label == 1]
        if two_indices:
            text = data['Text'][idx-2]
            for k in two_indices:
                text=text+f' <{id2label[k]}>'
        else:
            text = data['Text'][idx-2] + ' <NONE>'
        if one_indices:
            text=text+ sep+ data['Text'][idx-1]
            for k in one_indices:
                text=text+f' <{id2label[k]}>'
            text=text+sep+data['Text'][idx]
        else:
            text = text +sep+data['Text'][idx-1]+' <NONE>'+sep+data['Text'][idx]
        new_text.append(text)
    else:
        new_text.append(data['Text'][idx])
    data = data[['Text-ID', 'Sentence-ID', 'Text','language','pred_labels']][idx:idx+1]
    data['Text']=new_text
    data=datasets.Dataset.from_pandas(data)
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
#################################################################################################################################################################
#################################################################################################################################################################
trainer = transformers.Trainer(model=model)
predictions, label, metrics = trainer.predict(validation_dataset.select(range(1)), metric_key_prefix="predict")

check = thresholds(predictions)
validation_dataset=validation_dataset.to_pandas()
validation_dataset['pred_labels']=None
validation_dataset['pred_labels'][0]=check.tolist()[0]
validation_dataset=datasets.Dataset.from_pandas(validation_dataset)
#################################################################################################################################################################
#################################################################################################################################################################
temp=context_data_2(validation_dataset,1)
predictions, label, metrics = trainer.predict(temp, metric_key_prefix="predict")
check = thresholds(predictions)
validation_dataset=validation_dataset.to_pandas()
validation_dataset['pred_labels'][1]=check.tolist()[0]
validation_dataset=datasets.Dataset.from_pandas(validation_dataset)

for i in tqdm(range(2,len(validation_dataset))):
    temp=context_data_2(validation_dataset,i)
    predictions, label, metrics = trainer.predict(temp, metric_key_prefix="predict")
    check = thresholds(predictions)
    validation_dataset=validation_dataset.to_pandas()
    validation_dataset['pred_labels'][i]=check.tolist()[0]
    validation_dataset=datasets.Dataset.from_pandas(validation_dataset)

#################################################################################################################################################################
#################################################################################################################################################################
result =pd.DataFrame(validation_dataset['pred_labels'], columns=labels)
print(result)

result.to_dict('records')
df_prediction = pandas.DataFrame({'Text-ID':validation_dataset['Text-ID'],'Sentence-ID':validation_dataset['Sentence-ID']})                                                
df_prediction = pd.concat([df_prediction, pd.DataFrame.from_dict(result)], axis=1)

#################################################################################################################################################################
#########################################################   EXTRA CODE   ########################################################################################
#################################################################################################################################################################
# def transform_value(value):
#     if value == 1:
#         return 0.5
#     else:
#         return 0

# predictions = pandas.DataFrame()
# predictions['Text-ID'] = df_prediction['Text-ID']
# predictions['Sentence-ID'] = df_prediction['Sentence-ID']

# for col_name in labels:
#     predictions[f'{col_name} attained'] = df_prediction[col_name]   #.apply(lambda x: transform_value(x))
#     predictions[f'{col_name} constrained'] = 0                      #df_prediction[col_name].apply(lambda x: transform_value(x))

#################################################################################################################################################################
#################################################################################################################################################################

write_tsv_dataframe(os.path.join('', 'xlm-xl-layers-38_final.tsv'), df_prediction) #'EL_predictions.tsv' 

#################################################################################################################################################################
#################################################################################################################################################################
# predict_report = classification_report(y_true=validation_dataset['labels'], y_pred=validation_dataset['pred_labels'],
#                                                labels=range(19),
#                                                target_names=list(label2id.keys()), 
#                                                digits=4,
#                                                #average='micro'
#                                                )

# print(predict_report)

#################################################################################################################################################################
#################################################################################################################################################################
