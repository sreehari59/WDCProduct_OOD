"""
Run multi-class fine-tuning
"""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

from typing import Dict
import gc

from sklearn.manifold import TSNE


import pickle
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def aurra(confidence: torch.Tensor, correct: torch.Tensor) -> float:
    """
    Computes the area under the roc curve

    :param confidence: predicted confidence values
    :param correct: ground truth

    :return: aurroc score
    """
    conf_ranks = np.argsort(confidence.numpy())[::-1]  # indices from greatest to least confidence
    rra_curve = np.cumsum(np.asarray(correct.numpy())[conf_ranks])
    rra_curve = rra_curve / np.arange(1, len(rra_curve) + 1)  # accuracy at each response rate
    return float(np.mean(rra_curve))


def fpr_at_tpr(pred, target, k=0.95):
    """
    Calculate the False Positive Rate at a certain True Positive Rate

    :param pred: outlier scores
    :param target: target label
    :param k: cutoff value
    :return:
    """
    # results will be sorted in reverse order
    fpr, tpr, _ = binary_roc(pred, target)
    idx = torch.searchsorted(tpr, k)
    if idx == fpr.shape[0]:
        return fpr[idx - 1]

    return fpr[idx]

def fpr_95_tpr(pred, target, k=0.95):
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
    
class OODMetrics(object):

    def __init__(self, device: str = "cpu", mode: str = "classification"):
        super(OODMetrics, self).__init__()
        self.device = device
        self.buffer = TensorBuffer(device=self.device)

        if mode not in ["classification"]:
            raise ValueError("mode must be 'classification'")

        self.mode = mode

    def update(self, scores, y ):
        
        label = is_unknown(y).detach().to(self.device).long()
        scores = scores[0].detach().to(self.device)

        if self.mode == "classification":
            self.buffer.append("scores", scores)
            self.buffer.append("labels", label)

        return self

    def _compute(self, labels, scores): 

        if len(torch.unique(labels)) != 2:
            raise ValueError("Data must contain IN and OOD samples.")

        scores, scores_idx = torch.sort(scores, stable=True)
        labels = labels[scores_idx]
        
        auroc = binary_auroc(scores, labels)

        p, r, t = binary_precision_recall_curve(scores, labels)
        aupr_in = auc(r, p)

        p, r, t = binary_precision_recall_curve(-scores, 1 - labels)
        aupr_out = auc(r, p)

        fpr = fpr_at_tpr(scores, labels)

        return {"AUROC": auroc,  "AUPR-IN": aupr_in, "AUPR-OUT": aupr_out, "FPR95TPR": fpr,
        }

    def compute(self): 
        """
        Calculate metrics

        :return: dictionary with different metrics
        """
        if self.buffer.is_empty():
            raise ValueError("Must be given data to calculate metrics.")

        if self.mode == "classification":
            labels = self.buffer.get("labels").view(-1)
            scores = self.buffer.get("scores").view(-1)
            if len(torch.unique(labels)) != 2:
                raise ValueError("Data must contain IN and OOD samples.")
            metrics = self._compute(labels, scores)

        metrics = {k: v.item() for k, v in metrics.items()}
        return metrics

    def reset(self): 
        """
        Resets the buffer
        """
        self.buffer.clear()
        return self


class Mahalanobis(Detector):

    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        eps: float = 0.002,
        norm_std: Optional[List] = None,
    ):
        super(Mahalanobis, self).__init__()
        self.model = model
        self.mu: Tensor = None  #: Centers
        self.cov: Tensor = None  #: Covariance Matrix
        self.precision: Tensor = None  #: Precision Matrix
        self.eps: float = eps  #: epsilon
        self.norm_std = norm_std

    def fit(self: Self, data_loader: DataLoader, device: str = None) -> Self:
        """
        Fit parameters of the multi variate gaussian.

        :param data_loader: dataset to fit on.
        :param device: device to use
        :return:
        """
        if device is None:
            device = list(self.model.parameters())[0].device
            log.warning(f"No device given. Will use '{device}'.")

        z, y = extract_features(data_loader, self.model, device)
        return self.fit_features(z, y, device)

    def fit_features(self: Self, z: Tensor, y: Tensor, device: str = None) -> Self:
        """
        Fit parameters of the multi variate gaussian.

        :param z: features
        :param y: class labels
        :param device: device to use
        :return:
        """

        if device is None:
            device = list(self.model.parameters())[0].device
            log.warning(f"No device given. Will use '{device}'.")

        z, y = z.to(device), y.to(device)

        log.debug("Calculating mahalanobis parameters.")
        classes = y.unique()

        # we assume here that all class 0 >= labels <= classes.max() exist
        assert len(classes) == classes.max().item() + 1
        assert not contains_unknown(classes)

        n_classes = len(classes)
        # size of self.mu while taking the output of logits will be [500,500] n_classes 500 and if z is of shape [batchsize, 500] 
        # size of self.mu while taking the avg embedding will be [500,768] n_classes 500 and if z is of shape [batchsize, 768] 
        self.mu = torch.zeros(size=(n_classes, z.shape[-1])).to(device)  
        self.cov = torch.zeros(size=(z.shape[-1], z.shape[-1])).to(device)

        for clazz in range(n_classes):
            # It is element wise equality check. It will give a tensor with [True,False, True ,..] values. idx will have same shape of y
            idxs = y.eq(clazz)
            assert idxs.sum() != 0
            zs = z[idxs].to(device)
            self.mu[clazz] = zs.mean(dim=0)
            self.cov += (zs - self.mu[clazz]).T.mm(zs - self.mu[clazz])

        self.precision = torch.linalg.inv(self.cov)
        return self

    def predict_features(self, x: Tensor) -> Tensor:
        """
        Calculates mahalanobis distance directly on features.

        :param x: features, as given by the model.
        """
        features = x.view(x.size(0), x.size(1), -1) # say our x is [50, ]
        features = torch.mean(features, 2)
        noise_gaussian_scores = []
        mahal_dist_list = []
        for clazz in range(self.n_classes):
            
            centered_features = features.data - self.mu[clazz]
            term_gau = (
                -0.5
                * torch.mm(
                    torch.mm(centered_features, self.precision), centered_features.t()
                ).diag()
            )
            without_gau = torch.mm(torch.mm(centered_features, self.precision), centered_features.t()
                ).diag()

            noise_gaussian_scores.append(term_gau.view(-1, 1))
            mahal_dist_list.append(without_gau.view(-1, 1))

        noise_gaussian_score = torch.cat(noise_gaussian_scores, 1)
        noise_gaussian_score = torch.max(noise_gaussian_score, dim=1).values

        mahal_dist = torch.cat(mahal_dist_list, 1)

        return -noise_gaussian_score, mahal_dist

    def predict(self, x: Tensor, device: str = None) -> Tensor:
        """
        :param x: input tensor
        """
        if self.mu is None:
            raise RequiresFittingException

        if self.model is None:
            raise ModelNotSetException

        if self.eps > 0:
            x = self._odin_preprocess(x, x.device)

        # Considering the logits
        # output = self.model(x)
        # features = output.logits

        # Considering the average embedding size
        # output = self.model(x)
        # hidden_state = output.hidden_states 
        # layer_embeds = hidden_state[-1]
        # mask = torch.ones(layer_embeds.size(0), layer_embeds.size(2)).to(device)
        # features = torch.div(layer_embeds.sum(dim=1).to(0),mask.sum(dim=1,keepdim=True).to(0))

        # Considering the CLS token
        output = self.model(x)
        hidden_state = output.hidden_states 
        last_hidden_layer = hidden_state[-1]
        features = last_hidden_layer[:,0,:]

        return self.predict_features(features)


    @property
    def n_classes(self):
        """
        Number of classes the model is fitted for
        """
        if self.mu is None:
            raise RequiresFittingException

        return self.mu.shape[0]
    
class ToUnknown(object):
    """
    Callable that returns a negative number, used in pipelines to mark specific datasets as OOD or unknown.
    """

    def __init__(self):
        pass

    def __call__(self, y):
        return -1

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

def datasetToTuple(data_loader, max_length, attention_mask_flag):
    """
    param: 
        max_length is the max length of the tensor
        attention_mask_flag is set to True when we are conidering the average embedding
    """
    modified_dataloader = []
    for data_dict in data_loader:     
        if not attention_mask_flag:   
            # Remove the "attention_mask" key from the dictionary
            data_dict.pop("attention_mask", None)

        for key, value in data_dict.items():
            if key == 'input_ids':
                # Get the current length of the tensor
                current_length = value.size()[1]  
                if current_length < max_length:
                    padding_length = max_length - current_length                                      
                    padding = torch.ones(value.shape[0], padding_length)
                    padded_value = torch.cat((value, padding), dim=1)
                    data_dict[key] = padded_value     

            if attention_mask_flag and key == 'attention_mask': 
                # Get the current length of the tensor
                current_length = value.size()[1]
                if current_length < max_length:
                    padding_length = max_length - current_length                                      
                    padding = torch.zeros(value.shape[0], padding_length)
                    padded_value = torch.cat((value, padding), dim=1)
                    data_dict[key] = padded_value     
        
        for key, value in data_dict.items():
            if key == 'input_ids':
                data =  value.long()
            else:
                label = value.int()

            if attention_mask_flag and key == 'attention_mask': 
                mask =  value.long()

        if attention_mask_flag:
            my_tuple = (data, label, mask)
        else:
            my_tuple = (data, label)

        modified_dataloader.append(my_tuple)
    return modified_dataloader

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

def get_embeddings(model, x):
        output = model(x)
        hidden_state = output.hidden_states 
        last_hidden_layer = hidden_state[-1]
        return last_hidden_layer[:,0,:]

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
    plt.xlabel('Mahalanobis Distance Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.savefig(path+filename+".png")

def main():

    # Clearing memory in GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0'
    
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

    external_data = True
    contrastive_model = False
    dataset_name = "WDC Product Test Data"
    if external_data:
        dataset_name = "NewsGroup20 dataset"

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
    validation_dataset = raw_datasets["validation"]
    validation_dataset = BaselineMultiClassificationDataset(validation_dataset, dataset_type='validation', size=data_args.train_size, tokenizer=model_args.tokenizer,
                                                             dataset=data_args.dataset_name, additional_data=data_args.additional_data,
                                                               only_additional=data_args.only_additional, only_title=data_args.only_title)
    if external_data:
        test_dataset = raw_datasets["test"]
        test_dataset = BaselineMultiClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_title=data_args.only_title)
    else:    
        test_dataset = raw_datasets["ood_test"]
        test_dataset = BaselineOODClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer,
                                                       dataset=data_args.dataset_name, additional_data=data_args.additional_data,
                                                         only_additional=data_args.only_additional, only_title=data_args.only_title)

    data_collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer, padding='longest', max_length=256)

    # Making into batches using DataLoader
    loader_in_train = DataLoader(train_dataset, batch_size=50, shuffle=False,collate_fn=data_collator)
    loader_in_test = DataLoader(test_dataset, batch_size=50, shuffle=False,collate_fn=data_collator)

    if external_data:
        news_test = NewsGroup20Dataset("data/",tokenizer=model_args.tokenizer, max_length=256, download=True, train=True, ood = True)
        len_dataset = dataset_length(test_dataset, news_test)
        extData_loader_in_test = DataLoader([test_dataset[i] for i in range(0,len_dataset)] + news_test[:len_dataset],
                                              batch_size=50, shuffle=False,collate_fn=data_collator)
    
    # Formatting the dataloader
    
    # Set attention_mask_flag = True when considering average embedding else always False
    mahal_train = datasetToTuple(loader_in_train, max_length=256, attention_mask_flag = False)
    if external_data:
        mahal_test = datasetToTuple(extData_loader_in_test, max_length=256, attention_mask_flag = False)
    else:
        mahal_test = datasetToTuple(loader_in_test, max_length=256, attention_mask_flag = False)
    
    print(">>>Loading checkpoint")
    checkpoint_path = "../../reports/baseline-multi/wdcproductsmulti80cc20rnd000un-large-64-5e-05-roberta-base/"
    best_checkpoint_path = best_run(checkpoint_path)

    # Load the model configuration
    config = RobertaConfig.from_pretrained(best_checkpoint_path+"/config.json")
    config.output_hidden_states = True

    # Initialize the Roberta model
    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_path+"/pytorch_model.bin",config=config)
    model.to(device)
    
    
    print(">>>Fitting Mahalanobis distance")
    detectors = {}
    detectors["Mahalanobis"] = Mahalanobis(model, eps=0.0)
    for name, detector in detectors.items():
        detector.fit(mahal_train, device=device)
    print(">>>Fitted Mahalanobis distance on training data")
    
    mahalanobis_buffer = TensorBuffer()
    test_buffer = TensorBuffer()


    metrics = OODMetrics()
    results = []
    for detector_name, detector in detectors.items():   
        print(f">>>Evaluating {detector_name}")
        for x, y in mahal_test:
            metrics.update(detector(x.to(device)), y.to(device))
            mahalanobis_buffer.append("mahalanobis_dist_to_all", detector(x.to(device))[1])
            mahalanobis_buffer.append("mahal_score", detector(x.to(device))[0])
            mahalanobis_buffer.append("label", y)
            
            
            # test_buffer.append("embedding",get_embeddings(model, x.to(device)))
            # test_buffer.append("label",y)

            r = {"Detector": detector_name, "Dataset": dataset_name}

        r.update(metrics.compute())
        results.append(r)

    values = mahalanobis_buffer.get("mahalanobis_dist_to_all")
    mahal_score = mahalanobis_buffer.get("mahal_score")
    labels = mahalanobis_buffer.get("label")

    # new_scores, scores_idx = torch.sort(mahal_score.view(-1), stable=True)
    # labels = labels[scores_idx]
    label_list = labels.numpy()
    label_list = label_list * -1
    if external_data:
        labels = i.get("labels")
        labels = torch.where(labels >= 0, torch.tensor(0), torch.tensor(-1))
        labels = labels.numpy() + 1
    print("label_list: ",label_list)
    score_list = mahal_score.numpy()

    auroc = roc_auc_score(label_list, score_list)
    # In AUPR-Out the Out dist considered as positive class or label 1
    aupr_out = average_precision_score(label_list, score_list)
    # In AUPR-In the In dist considered as positive class so changing the lable
    aupr_in = average_precision_score(-1 * label_list + 1, 1 - score_list)
    fpr_tpr = fpr_95_tpr(score_list, label_list)

    print("AUROC (higher is better): ", round(auroc*100, 2))
    print("AUPR_IN (higher is better): ", round(aupr_in*100, 2))
    print("AUPR_OUT (higher is better): ", round(aupr_out*100, 2))
    print("FPR@TPR 95 (lower is better): ",round(fpr_tpr, 2))
    print(results)
    print(">>>best_checkpoint_path: ",best_checkpoint_path)
    print(">>>Model Trained on dataset:",data_args.train_file.split("/")[-1])
    print(">>>Model detecting OOD on dataset:",data_args.ood_test_file.split("/")[-1])


    # Uncomment this part to visualize the plots
    
    # if contrastive_model:
    #     model_name= "Contrastive"
    # else:
    #     model_name = "Roberta-base"

    # if external_data:
    #     auroc_file_name = "AUROC_WDC_News" + model_name
    #     plot_file_name = "Prob_Density_WDC_News_" + model_name

    #     auroc_plot(label_list,  score_list,"AUROC WDC Product & NewsGroup20 Distance Approach",
    #                 auroc_file_name, data_args)
    #     plot_density(label_list,  score_list, "WDC Product & NewsGroup20 Distance Approach",
    #                     plot_file_name, data_args)
    # else:
    #     dataset_name = raw_datasets["ood_test"]
    #     if "valid" in dataset_name:
    #         dataset_name = dataset_name.split("/")[-1].split(".")[0]
    #     else:
    #         dataset_name = dataset_name.split("/")[-1].split("_")[1]

    #     auroc_file_name = "AUROC_WDC_" + dataset_name + "_" +  model_name
    #     plot_file_name = "Prob_Density_WDC_" + dataset_name + "_" + model_name

    #     auroc_plot(label_list,  score_list,"AUROC "+dataset_name+" Distance Approach",
    #                 auroc_file_name, data_args)
    #     plot_density(label_list,  score_list, dataset_name+" Distance Approach",
    #                     plot_file_name, data_args)


if __name__ == "__main__":
    # The first dataset is the dataset on which the model was trained and the second the ood detection test file
    # CUDA_VISIBLE_DEVICES=1 bash lspc/distance_based.sh roberta-base True 64 5e-05 wdcproductsmulti20cc80rnd000un large wdcproductsmulti20cc80rnd050un_gs.pkl
    main()


