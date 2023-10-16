"""
Run multi-class fine-tuning
References: https://www.youtube.com/watch?v=u--UVvH-LIQ
"""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
np.random.seed(42)
import random
random.seed(42)

import pandas as pd
from sklearn.metrics import classification_report

import logging
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoConfig

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

from src.contrastive.data.datasets import BaselineMultiClassificationDataset
from src.contrastive.models.metrics import compute_metrics_baseline_multiclass

from transformers import EarlyStoppingCallback

from transformers.utils.hp_naming import TrialShortNamer

from pdb import set_trace

from torch.autograd import Variable
import gc
os.environ["WANDB_DISABLED"] = "true"
from torch.utils.data import DataLoader
from sklearn import metrics as sklearnmetric
from src.contrastive.data.datasets import BaselineOODClassificationDataset, NewsGroup20Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional.classification import binary_roc

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.2")

logger = logging.getLogger(__name__)

#MODEL_PARAMS=['frozen', 'pool', 'use_colcls', 'sum_axial']

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


def dataset_length(dataset1, dataset2):
    """
    When 2 datasets given returns the dataset with the smallest length
    """
    dataset_length = len(dataset1)
    if dataset_length > len(dataset2):
        dataset_length = len(dataset2)
    return dataset_length
        
def encode_onehot(labels, n_classes):
    onehot = torch.FloatTensor(labels.size()[0], n_classes)
    labels = labels.data
    if labels.is_cuda:
        onehot = onehot.cuda()
    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot

budget = 0.7
lmbda = 0.3

# Custom trainer inherits the Trainer class
class CustomTrainer(Trainer):
    # This compute loss is overriding the compute loss inside the Trainer
    def compute_loss(self, model, inputs, return_outputs=False):
        global budget
        global lmbda
        labels = inputs.get("labels")
        labels_onehot = Variable(encode_onehot(labels, 500))
        confidence, pred_original = model(**inputs)
        # pred_original = pred[:,1:].cuda()
        # print("pred shape: ",pred.shape)
        # print("pred_original shape: ",pred_original.shape)

        # Compute class prediction probabilities p using softmax
        pred_original = torch.nn.functional.softmax(pred_original, dim=-1)
        confidence = torch.sigmoid(confidence)
        # confidence = confidence / torch.sqrt(torch.tensor(500.0))

        eps = 1e-12
        pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
        confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

        # Setting p_success = 0.3 means it will create b which will have 30% chance of having 1
        # and 70% chance of having 0. 70% of the time the model is not given any hints
        p_success = 0.5
        b = torch.bernoulli(torch.Tensor(confidence.size()).fill_(p_success)).cuda()
        conf = confidence * b + (1 - b)

        pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot  * (1 - conf.expand_as(labels_onehot))
        print("confidence values:",confidence.view(-1))

        pred_new = torch.log(pred_new)
        
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.NLLLoss(weight=torch.tensor(self.pos_neg).to(self.model.device))
        task_loss = loss_fct(pred_new.view(-1, self.model.num_labels), labels.view(-1))
        # Calculate the confidence loss
        # Confidence loss high means taking more hints
        confidence_loss = torch.mean(-torch.log(confidence))

        total_loss = task_loss + (lmbda * confidence_loss)
        print("total_loss: ",total_loss)
        print("task_loss: ",task_loss)
        print("confidence_loss: ",confidence_loss)
        print("lmbda: ",lmbda)
        
        if budget > confidence_loss.item():
                lmbda = lmbda / 1.01
        # If confidence loss is greater than budget then increase lambda
        # that is make it more expensive to ask for hints
        # The confidence loss becomes more hints are asked
        elif budget <= confidence_loss.item():
                lmbda = lmbda / 0.99

        return (total_loss, pred_original) if return_outputs else total_loss
    

class CustomModel(nn.Module):
    def __init__(self, model_args, num_labels, train_dataset,device):
        super(CustomModel, self).__init__()
        # self.config = AutoConfig.from_pretrained(model_args.tokenizer)
        # self.base_model = AutoModelForSequenceClassification.from_pretrained(model_args.tokenizer, config=self.config)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(model_args.tokenizer, num_labels=num_labels)
        self.num_labels = num_labels
        if model_args.grad_checkpoint:
                self.base_model._set_gradient_checkpointing(self.base_model, True)
        self.linear1 = nn.Linear(num_labels, 128)
        self.bm = nn.BatchNorm1d(128, affine=False)
        self.linear2 = nn.Linear(128, 1)
        self.train_dataset = train_dataset
        self.device = device
        self.base_model.resize_token_embeddings(len(self.train_dataset.tokenizer))
    
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.base_model(input_ids=input_ids,
                         attention_mask=attention_mask, 
                         output_hidden_states=False)
        pred = outputs.logits
        confidence = self.linear1(outputs.logits)
        if confidence.size(0) == 1:
            self.bm.eval()
        confidence = self.bm(confidence)
        confidence = self.linear2(confidence)
        return (confidence, pred)

def auroc_plot(actual, predicted,title,filename,data_args):
    path = data_args.plot_path
    fpr, tpr, _ = sklearnmetric.roc_curve(actual,  predicted)
    auroc = sklearnmetric.roc_auc_score(actual, predicted)
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
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
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


def main():

    ood_detection = False
    external_data = False

    # Clearing memory in GPU
    gc.collect()
    torch.cuda.empty_cache()

    
    def get_posneg(train_dataset):
        # Train dataset is multi class so say label 1 might have 3 rows of data but label 2 might have 4 rows of data and so on
        # So here intuitively we are saying assign high weight to rare classes and low weight to common classes
        counts = train_dataset.data['label'].value_counts()
        max_count = max(counts.values)
        counts = max_count / counts
        counts = counts.apply(math.ceil)
        # Returns an array
        return counts.values.astype('float16')
    

    # HfArgumentParser and TrainingArguments imported from transformers
    # ModelArguments and DataTrainingArguments are two classes
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

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

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

    logger.info(f"raw_datasets {raw_datasets}")
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        logger.info(f"train_dataset {train_dataset}")
        train_dataset = BaselineMultiClassificationDataset(train_dataset, dataset_type='train', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, aug=data_args.augment, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_title=data_args.only_title)
        if training_args.evaluation_strategy != 'no':
            validation_dataset = raw_datasets["validation"]
            logger.info(f"validation_dataset {validation_dataset}")
            validation_dataset = BaselineMultiClassificationDataset(validation_dataset, dataset_type='validation', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_title=data_args.only_title)
        if training_args.load_best_model_at_end:
            test_dataset = raw_datasets["test"]
            logger.info(f"test_dataset {test_dataset}")
            test_dataset = BaselineMultiClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_title=data_args.only_title)
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        validation_dataset = raw_datasets["validation"]
        logger.info(f"elif validation_dataset {validation_dataset}")
        validation_dataset = BaselineMultiClassificationDataset(validation_dataset, dataset_type='validation', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_title=data_args.only_title)

    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        logger.info(f"elif test_dataset {test_dataset}")
        test_dataset = BaselineMultiClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, additional_data=data_args.additional_data, only_additional=data_args.only_additional, only_title=data_args.only_title)

    num_labels = len(test_dataset.data['label'].unique())
    logger.info(f"num_labels {num_labels}")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=train_dataset.tokenizer, padding='longest', max_length=256)

    # Early stopping callback
    callback = EarlyStoppingCallback(early_stopping_patience=20)

    output_dir = deepcopy(training_args.output_dir)
    
    if not ood_detection:
        for run in range(3):
            init_args = {}

            training_args.save_total_limit = 1
            training_args.seed = run
            training_args.output_dir = f'{output_dir}{run}'
            

            # Detecting last checkpoint.
            last_checkpoint = None
            if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
                last_checkpoint = get_last_checkpoint(training_args.output_dir)

            pos_neg = get_posneg(train_dataset)
            model = CustomModel(model_args,num_labels, train_dataset,training_args.device)

            # Initialize our Trainer
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=validation_dataset if training_args.do_eval else None,
                data_collator=data_collator,
                # optimizers=(optimizer, scheduler),
                compute_metrics=compute_metrics_baseline_multiclass,
                callbacks=[callback]
            )
            trainer.pos_neg = get_posneg(train_dataset)
            # Training
            if training_args.do_train:

                time.sleep(30)
                # Train the model from the checkpoint if the checkpoint exist
                # train_result = trainer.train(resume_from_checkpoint=checkpoint)
                train_result = trainer.train()
                trainer.save_model()  # Saves the tokenizer too for easy upload

                metrics = train_result.metrics
                max_train_samples = (
                    data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
                )
                metrics["train_samples"] = min(max_train_samples, len(train_dataset))

                trainer.log_metrics(f"train", metrics)
                trainer.save_metrics(f"train", metrics)
                trainer.save_state()
            
            # Evaluation
            results = {}
            if training_args.do_eval:
                logger.info("*** Evaluate ***")

                metrics = trainer.evaluate(
                    metric_key_prefix="eval"
                )
                max_eval_samples = len(validation_dataset)
                metrics["eval_samples"] = max_eval_samples

                trainer.log_metrics(f"eval", metrics)
                trainer.save_metrics(f"eval", metrics)

            if training_args.do_predict or training_args.do_train:
                logger.info("*** Predict ***")

                predict_results = trainer.predict(
                    test_dataset,
                    metric_key_prefix="predict"
                )

                metrics = predict_results.metrics
                max_predict_samples = len(test_dataset)
                metrics["predict_samples"] = max_predict_samples

                trainer.log_metrics(f"predict", metrics)
                trainer.save_metrics(f"predict", metrics)
        
    else:
        results = {}
        if not external_data:
            test_dataset = raw_datasets["ood_test"]
            test_dataset = BaselineOODClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer,
                                                        dataset=data_args.dataset_name, additional_data=data_args.additional_data,
                                                            only_additional=data_args.only_additional, only_title=data_args.only_title)
            loader_in_test = DataLoader(test_dataset, batch_size=50, shuffle=False,collate_fn=data_collator)

        else:
            news_test = NewsGroup20Dataset("data/",tokenizer=model_args.tokenizer, max_length=256, download=True, train=True, ood = True)
            len_dataset = dataset_length(test_dataset, news_test)
            loader_in_test = DataLoader([test_dataset[i] for i in range(0,len_dataset)] + news_test[:len_dataset],
                                                batch_size=50, shuffle=False,collate_fn=data_collator)
            
        print(">>>Loading checkpoint")
        checkpoint_path = deepcopy(training_args.output_dir)
        best_checkpoint_path = best_run(checkpoint_path)
        model_path = best_checkpoint_path+"/pytorch_model.bin"
        model = CustomModel(model_args,num_labels, train_dataset, training_args.device)
        model.load_state_dict(torch.load(model_path, map_location=training_args.device))

        # in distribution is 1 and out of distribution is 0
        label_list = []
        score_list = []
        for i in loader_in_test:
            confidence, _  = model(i['input_ids'], i['attention_mask'],i['labels'])
            confidence = torch.sigmoid(confidence)
            confidence = confidence.squeeze(-1)
            
            scores = confidence.data.cpu().numpy() #output will be a numpy array
            scores = 1 - scores 
            score_list.append(scores)

            if external_data:
                labels = i.get("labels")
                labels = torch.where(labels >= 0, torch.tensor(0), torch.tensor(-1))
                labels = labels.numpy() + 1
            else:
                # labels = i.get("labels").numpy() + 1
                labels = i.get("labels").numpy() * -1
            label_list.append(labels)
        
        score_list = np.concatenate(score_list)
        label_list = np.concatenate(label_list)
        print("confidence score_list: ",score_list)
        print("label_list: ",label_list)
        auroc = sklearnmetric.roc_auc_score(label_list, score_list)
        aupr_in = sklearnmetric.average_precision_score(label_list, score_list)
        aupr_out = sklearnmetric.average_precision_score(-1 * label_list + 1, 1 - score_list)
        fpr_tpr = fpr_at_tpr(score_list, label_list)
        print("AUROC (higher is better): ", auroc)
        print("AUPR_IN (higher is better): ", aupr_in)
        print("AUPR_OUT (higher is better): ", aupr_out)
        print("FPR@TPR 95 (lower is better): ",round(fpr_tpr, 2))

        print(">>>best_checkpoint_path: ",best_checkpoint_path)
        print(">>>Model Trained on dataset:",data_args.train_file.split("/")[-1])
        print(">>>Model detecting OOD on dataset:",data_args.ood_test_file.split("/")[-1])

        # Back bone model
        # model_name = "Roberta-base"
        # if external_data:
        #     auroc_file_name = "AUROC_WDC_News" + model_name
        #     plot_file_name = "Prob_Density_WDC_News_" + model_name

        #     auroc_plot(label_list,  score_list,"AUROC WDC Product & NewsGroup20 Classification Approach",
        #                 auroc_file_name, data_args)
        #     plot_density(label_list,  score_list, "WDC Product & NewsGroup20 Classification Approach",
        #                     plot_file_name, data_args)
        # else:
        #     dataset_name = raw_datasets["ood_test"]
        #     if "valid" in dataset_name:
        #         dataset_name = dataset_name.split("/")[-1].split(".")[0]
        #     else:
        #         dataset_name = dataset_name.split("/")[-1].split("_")[1]

        #     auroc_file_name = "AUROC_WDC_" + dataset_name + "_" +  model_name
        #     plot_file_name = "Prob_Density_WDC_" + dataset_name + "_" + model_name

        #     auroc_plot(label_list,  score_list,"AUROC "+dataset_name+" Classification Approach",
        #                 auroc_file_name, data_args)
        #     plot_density(label_list,  score_list, dataset_name+" Classification Approach",
        #                     plot_file_name, data_args)
            

    return results

if __name__ == "__main__":
    main()

#CUDA_VISIBLE_DEVICES=3 bash lspc/classification_based.sh roberta-base True 64 5e-05 wdcproductsmulti80cc20rnd000un large wdcproductsmulti20cc80rnd050un_gs.pkl



