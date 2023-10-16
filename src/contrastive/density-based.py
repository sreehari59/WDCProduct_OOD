"""
Run multi-class fine-tuning
"""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from torch.utils.data import DataLoader
import torch
from transformers import RobertaModel, RobertaConfig
from transformers import AutoModelForSequenceClassification

from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_roc,
)
from torchmetrics.utilities.compute import auc

from utils import TensorBuffer, is_unknown


import numpy as np
np.random.seed(42)
import random
random.seed(42)
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import time
import math

from copy import deepcopy

import torch
from torch import nn

import transformers as transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from src.contrastive.data.datasets import BaselineMultiClassificationDataset, BaselineOODClassificationDataset, NewsGroup20Dataset
from src.contrastive.models.metrics import compute_metrics_baseline_multiclass

from transformers import EarlyStoppingCallback

from transformers.utils.hp_naming import TrialShortNamer

from pdb import set_trace

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.2")

logger = logging.getLogger(__name__)

#MODEL_PARAMS=['frozen', 'pool', 'use_colcls', 'sum_axial']

import logging
import warnings
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from typing import TypeVar
from typing import Callable, List, Optional, TypeVar
from api import Detector, ModelNotSetException, RequiresFittingException
from utils import TensorBuffer, contains_unknown, extract_features, is_known, is_unknown

log = logging.getLogger(__name__)

Self = TypeVar("Self")
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from typing import Dict
import gc

from sklearn.manifold import TSNE


import pickle
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
from sklearn.metrics import average_precision_score,average_precision_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from torch import logsumexp
from src.contrastive.models.modeling import ContrastiveClassifierModelMultiForOOD
from torchmetrics.functional.classification import binary_auroc

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_pretrained_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    do_param_opt: Optional[bool] = field(
        default=False, metadata={"help": "If aou want to do hyperparamter optimization"}
    )
    frozen: Optional[str] = field(
        default='frozen', metadata={"help": "If encoder params should be frozen, options: frozen, unfrozen"}
    )
    grad_checkpoint: Optional[bool] = field(
        default=True, metadata={"help": "If aou want to use gradient checkpointing"}
    )
    tokenizer: Optional[str] = field(
        default='huawei-noah/TinyBERT_General_4L_312D',
        metadata={
            "help": "Tokenizer to use"
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    train_size: Optional[str] = field(
        default=None, metadata={"help": "The size of the training set."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    augment: Optional[str] = field(
        default=None, metadata={"help": "The data augmentation to use."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    dataset_name: Optional[str] = field(
        default='lspc',
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    additional_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to additional data to be used for training"
        },
    )
    only_additional: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If only additional data should be used without domain training data"
        },
    )
    only_title: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use only the title attribute"
        },
    )
    ood_test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input OOD test data file for out of distribution detection."
        },
    )
    plot_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional path to save the plots."
        },
    )
    dataframe_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional path to save the results(confidence score) of the model."
        },
    )
    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need a training file.")

def best_run(checkpoint_path):  
    """
    Takes the best model given the checkpoint path.
    
    :param checkpoint_path: relative path to the checkpoint model.
    """       
    dir_list = os.listdir(checkpoint_path) 
    best_test_accuracy = 0.0
    best_run = 0
    for i in dir_list:
        if i.isdigit():
            file_name = 'all_results.json'
            file_path = checkpoint_path + i + '/' + file_name
            json_file = open(file_path)
            file_data = json.load(json_file)

            if best_test_accuracy < float(file_data.get('predict_accuracy')):
                best_test_accuracy = float(file_data.get('predict_accuracy'))
                best_run = i
    
    best_checkpoint_path = checkpoint_path + best_run
    return best_checkpoint_path


def dataset_length(dataset1, dataset2):
    """
    When 2 datasets given returns the dataset with the smallest length
    """
    dataset_length = len(dataset1)
    if dataset_length > len(dataset2):
        dataset_length = len(dataset2)
    return dataset_length

def result(actual,predicted):
    cm = confusion_matrix(actual,predicted)
    plt.figure().set_figwidth(0.5)
    plt.figure().set_figheight(3)
    sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['In-Dist','Out-Dist'],
            yticklabels=['In-Dist','Out-Dist'])
    plt.ylabel('Prediction',fontsize=10)
    plt.xlabel('Actual',fontsize=10)
    plt.title('Confusion Matrix',fontsize=15)
    
    plt.show()

    acc = accuracy_score(actual,predicted)
    precision = precision_score(actual,predicted)
    recall = recall_score(actual,predicted)
    f1 = f1_score(actual,predicted)
    print("Accuracy : ",acc)
    print("Precision : ",precision)
    print("Recall : ",recall)
    print("f1 : ",f1)

def energy_score(logits: Tensor, temp: Optional[float] = 1.0) -> Tensor:
        """
        :param logits: logits of input
        :param t: temperature value
        """
        return -temp * logsumexp(logits / temp, dim=1)

def auroc_plot(actual, predicted,title,filename,data_args):
    path = data_args.plot_path
    fpr, tpr, _ = roc_curve(actual,  predicted)
    auroc = roc_auc_score(actual, predicted)
    plt.clf()
    plt.plot(fpr,tpr, color="orange",label="AUC="+str(auroc))
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title)
    plt.legend(loc=4)

    if not os.path.isdir(data_args.plot_path):
        os.makedirs(data_args.plot_path)

    plt.savefig(path+filename+".png")


def plot_density(actual, predicted, title, filename, data_args, bins=50,):
    path = data_args.plot_path
    df_path = data_args.dataframe_path
    plot_df = pd.DataFrame({"actual": actual,
                           "predicted": predicted})
    
    if not os.path.isdir(data_args.dataframe_path):
        os.makedirs(data_args.dataframe_path)

    plot_df.to_csv(df_path+filename+".csv")
    plt.clf()
    # Plot histograms for positive and negative classes
    plt.hist(plot_df[plot_df["actual"]==0]["predicted"].values, bins=bins, label='In Distribution',
              alpha=0.5, color='blue')
    plt.hist(plot_df[plot_df["actual"]==1]["predicted"].values, bins=bins, label='Out of Distribution',
              alpha=0.5, color='red')

    # Set labels and legend
    plt.xlabel('Negative Energy Score')
    plt.ylabel('Frequency')
    plt.gca().invert_xaxis()
    plt.title(title)
    plt.legend()
    plt.savefig(path+filename+".png")

def fpr_at_tpr(pred, target, k=0.95):
    """
    Calculate the False Positive Rate at a certain True Positive Rate

    :param pred: predicted energy score
    :param target: actual label
    :param k: cutoff value
    :return:
    """
    if isinstance(target, np.ndarray):
        target = torch.tensor(target, dtype=torch.int)

    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)

    # results will be sorted in reverse order
    fpr, tpr, _ = binary_roc(pred, target)
    idx = torch.searchsorted(tpr, k)
    if idx == fpr.shape[0]:
        return fpr[idx - 1]

    return fpr[idx].item() * 100

def cls_embed_getter(model_output):
    hidden_state = model_output.hidden_states 
    # last_hidden_layer is of size torch.Size([50, 256, 768]) when batch size is 50
    last_hidden_layer = hidden_state[-1] 
    # Taking the CLS token 
    cls_token = last_hidden_layer[:,0,:]
    return cls_token.detach().cpu().numpy()

def get_cls_embeddings(model, input):
        output = model(input['input_ids'], input['attention_mask'])
        hidden_state = output.hidden_states 
        last_hidden_layer = hidden_state[-1]
        return last_hidden_layer[:,0,:]

def cls_embed_df(cls_token_buffer,data_args,filename):
    if not os.path.isdir(data_args.dataframe_path):
        os.makedirs(data_args.dataframe_path)

    df_path = data_args.dataframe_path
    plot_df = pd.DataFrame(cls_token_buffer.get("cls_embedding").tolist())
    plot_df['label'] = cls_token_buffer.get("label").tolist()
    plot_df.to_csv(df_path+filename+".csv")

# def cls_embed_df(cls_token_list,label_list,data_args,filename):
#     df_path = data_args.dataframe_path
#     plot_df = pd.DataFrame({"label": label_list,
#                            "CLS": cls_token_list})
#     plot_df.to_csv(df_path+filename+".csv")

def main():
    init_args = {}
    def get_posneg(train_dataset):
        counts = train_dataset.data['label'].value_counts()
        max_count = max(counts.values)
        counts = max_count / counts
        counts = counts.apply(math.ceil)
        return counts.values.astype('float16')

    # Clearing memory in GPU
    gc.collect()
    torch.cuda.empty_cache()
    
    external_data = False
    contrastive_model = True
    dataset_name = "NewsGroup20 dataset"
    if external_data:
        dataset_name = "WDC Product Test Data"
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    if data_args.ood_test_file is not None:
        data_files["ood_test"] = data_args.ood_test_file
    raw_datasets = data_files

    # Loading the datasets
    train_dataset = raw_datasets["train"]
    train_dataset = BaselineMultiClassificationDataset(train_dataset, dataset_type='train', size=data_args.train_size, tokenizer=model_args.tokenizer,
                                                        dataset=data_args.dataset_name, aug=data_args.augment, additional_data=data_args.additional_data,
                                                          only_additional=data_args.only_additional, only_title=data_args.only_title)
    test_dataset = raw_datasets["ood_test"]
    print("test_dataset: ",test_dataset)
    test_dataset = BaselineOODClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer,
                                                       dataset=data_args.dataset_name, additional_data=data_args.additional_data,
                                                         only_additional=data_args.only_additional, only_title=data_args.only_title)

    data_collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer, padding='longest', max_length=256)

    print(">>>Loading checkpoint")
    if contrastive_model:
        checkpoint_path = "../../reports/contrastive-ft-multi/wdcproductsmulti80cc20rnd000un-large-1024-5e-05-0.07-frozen-roberta-base/"
        best_checkpoint_path = best_run(checkpoint_path)
        model_path = best_checkpoint_path+"/pytorch_model.bin"

        # Loading the test data to get the number of labels
        test_dataset = raw_datasets["test"]
        test_dataset = BaselineMultiClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_title=data_args.only_title)
        num_labels = len(test_dataset.data['label'].unique())

        pos_neg = get_posneg(train_dataset)
        model = ContrastiveClassifierModelMultiForOOD(checkpoint_path=model_path, len_tokenizer=len(train_dataset.tokenizer),
                                                  num_labels=num_labels, model=model_args.tokenizer, frozen=model_args.frozen,
                                                    pos_neg=pos_neg, **init_args)
        # print(model)
    else:
        checkpoint_path = deepcopy(training_args.output_dir)
        # "../../reports/baseline-multi/wdcproductsmulti80cc20rnd000un-large-64-5e-05-roberta-base/"
        best_checkpoint_path = best_run(checkpoint_path)

        # Load the model configuration
        config = RobertaConfig.from_pretrained(best_checkpoint_path+"/config.json")
        config.output_hidden_states = True

        # Initialize the Roberta model
        model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_path+"/pytorch_model.bin",config=config)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)
    
    print(">>>Evaluating")
    if not external_data:
        test_dataset = raw_datasets["ood_test"]
        test_dataset = BaselineOODClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer,
                                                    dataset=data_args.dataset_name, additional_data=data_args.additional_data,
                                                        only_additional=data_args.only_additional, only_title=data_args.only_title)
        loader_in_test = DataLoader(test_dataset, batch_size=50, shuffle=False,collate_fn=data_collator)

    else:
        news_test = NewsGroup20Dataset("data/",tokenizer=model_args.tokenizer, max_length=256, download=True, train=True, ood = True)
        len_dataset = dataset_length(test_dataset, news_test)
        # The test_dataset used here is the dataset from BaselineMultiClassificationDataset
        # it is the data which is having all in-distribution data but the 
        # model has not seen before
        loader_in_test = DataLoader([test_dataset[i] for i in range(0,len_dataset)] + news_test[:len_dataset],
                                            batch_size=50, shuffle=False,collate_fn=data_collator)
    
    
    
    label_list = []
    score_list = []
    cls_token_buffer = TensorBuffer()

    for i in loader_in_test:
        if contrastive_model:
            output = model(i['input_ids'], i['attention_mask'], i['labels'])
            logits = output[1]
            # cls_embedding = output[2]
        else:
            output = model(i['input_ids'], i['attention_mask'])
            logits = output.logits
            cls_embedding = get_cls_embeddings(model, i)

        energy_scores = energy_score(logits)
        score_list.append(energy_scores.data.cpu().numpy())

        if external_data:
            labels = i.get("labels")
            labels = torch.where(labels >= 0, torch.tensor(0), torch.tensor(-1))
            labels = labels.numpy() + 1
        else:
            labels = i.get("labels").numpy() + 1

        label_list.append(labels)

        # cls_token_buffer.append("cls_embedding",cls_embedding)
        # cls_token_buffer.append("label",i['labels'])
        
        # cls_token_list.append(cls_token)

    score_list = np.concatenate(score_list)
    label_list = np.concatenate(label_list)

    index = 0
    for i in label_list:    
        if i==1:
            label_list[index] = 0
        else:
            label_list[index] = 1
        index = index + 1

    # in distribution is 0 and out of distribution is 1
    print("energy score_list: ",score_list)
    print("label_list: ",label_list)
    auroc = roc_auc_score(label_list, score_list)
    # In AUPR-Out the Out dist considered as positive class or label 1
    aupr_out = average_precision_score(label_list, score_list)
    # In AUPR-In the In dist considered as positive class so changing the lable
    aupr_in = average_precision_score(-1 * label_list + 1, 1 - score_list)
    fpr_tpr = fpr_at_tpr(score_list, label_list)

    t = torch.tensor(label_list, dtype=torch.int)
    p = torch.tensor(score_list)
    print("binary_auroc: ", binary_auroc(p, t))
    print("AUROC (higher is better): ", round(auroc*100, 2))
    print("AUPR_IN (higher is better): ", round(aupr_in*100, 2))
    print("AUPR_OUT (higher is better): ", round(aupr_out*100, 2))
    print("FPR@TPR 95 (lower is better): ",round(fpr_tpr, 2))
    
    print(">>>best_checkpoint_path: ",best_checkpoint_path)
    print(">>>Model Trained on dataset:",data_args.train_file.split("/")[-1])
    print(">>>Model detecting OOD on dataset:",data_args.ood_test_file.split("/")[-1])
    print()

    if contrastive_model:
        model_name= "Contrastive"
    else:
        model_name = "Roberta-base"

    # cls_embed_df(cls_token_buffer,data_args,"CLS_token_"+model_name)

    
    if external_data:
        auroc_file_name = "AUROC_WDC_News" + model_name
        plot_file_name = "Prob_Density_WDC_News_" + model_name

        auroc_plot(label_list,  score_list,"AUROC WDC Product & NewsGroup20 Density Approach",
                    auroc_file_name, data_args)
        plot_density(label_list,  score_list, "WDC Product & NewsGroup20 Density Approach",
                        plot_file_name, data_args)
    else:
        dataset_name = raw_datasets["ood_test"]
        if "valid" in dataset_name:
            dataset_name = dataset_name.split("/")[-1].split(".")[0]
        else:
            dataset_name = dataset_name.split("/")[-1].split("_")[1]

        auroc_file_name = "AUROC_WDC_" + dataset_name + "_" +  model_name
        plot_file_name = "Prob_Density_WDC_" + dataset_name + "_" + model_name

        auroc_plot(label_list,  score_list,"AUROC "+dataset_name+" Density Approach",
                    auroc_file_name, data_args)
        plot_density(label_list,  score_list, dataset_name+" Density Approach",
                        plot_file_name, data_args)



if __name__ == "__main__":
    # The first dataset is the dataset on which the model was trained and the second the ood detection test file
    # CUDA_VISIBLE_DEVICES=1 bash lspc/density_based.sh roberta-base True 64 5e-05 wdcproductsmulti20cc80rnd000un large wdcproductsmulti20cc80rnd050un_gs.pkl
    main()


