import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

data_path = '../data/us-patent-phrase-to-phrase-matching/'
train_file = 'train.csv'
test_file = 'test.csv'

df_train = pd.read_csv(data_path + train_file)
df_test = pd.read_csv(data_path + test_file)
cpc_codes = pd.read_csv('../data/cooperative-patent-classification-codes-meaning/titles.csv').rename(columns={"code" : "context"})

cpc_codes = cpc_codes.rename(columns = {"code" : "context"})
df_train = pd.merge(df_train, cpc_codes[["context","title"]], on ="context", how = "left")
df_test = pd.merge(df_test, cpc_codes[["context","title"]], on ="context", how = "left")

from transformers import (PreTrainedModel, RobertaModel, RobertaTokenizerFast, RobertaConfig,
                          get_constant_schedule_with_warmup, AdamW, RobertaTokenizer, BertTokenizerFast)
from torch.utils.data import DataLoader, Dataset

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


MAX_ANCHOR_LEN = 40
MAX_TARGET_LEN = 50
MAX_TITLE_LEN = 175

df_train['input'] = df_train.apply(lambda x: (x.anchor + ' | ' + x.target, x.title), axis=1)  # Not sure '|' is a good idea
df_test['input'] = df_test.apply(lambda x: (x.anchor + ' | ' + x.target, x.title), axis=1)  # Not sure '|' is a good idea

df_train['out'] = pd.get_dummies(df_train.score, prefix='score').agg(list, axis=1)


from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df_train, test_size=.05, shuffle=True, random_state=41)

class PatentDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_anchor_len, max_target_len, max_title_len):
        super(PatentDataset, self).__init__()
        self.tokenizer = tokenizer
        self.df = dataset
        self.max_length = max_anchor_len + max_target_len + max_title_len  # FIXME

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        seq = self.df.input.iloc[index]

        inputs = self.tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            padding='max_length'
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.df.out.iloc[index], dtype=torch.float)
        }

train_dataset = PatentDataset(tokenizer=tokenizer, dataset=df_train, max_anchor_len=MAX_ANCHOR_LEN, max_target_len=MAX_TARGET_LEN, max_title_len=MAX_TITLE_LEN)
val_dataset = PatentDataset(tokenizer=tokenizer, dataset=df_val, max_anchor_len=MAX_ANCHOR_LEN, max_target_len=MAX_TARGET_LEN, max_title_len=MAX_TITLE_LEN)
test_dataset = PatentDataset(tokenizer=tokenizer, dataset=df_test, max_anchor_len=MAX_ANCHOR_LEN, max_target_len=MAX_TARGET_LEN, max_title_len=MAX_TITLE_LEN)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, num_workers=12)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=64, num_workers=12)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, num_workers=12, shuffle=False)


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch import nn
from transformers import (PreTrainedModel, RobertaModel, RobertaTokenizerFast, RobertaConfig,
                          get_constant_schedule_with_warmup, AdamW, RobertaTokenizer, BertTokenizerFast)

from transformers import BertModel
import pytorch_lightning as pl
import torch.nn.functional as F

seed_everything(42)


use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
print('Using device', device)


from pytorch_lightning.utilities.types import STEP_OUTPUT


class BERT(pl.LightningModule):
    def __init__(self, num_classes):
        super(BERT, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        out = torch.sigmoid(self.out(o2))
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, 'val')

    def _common_step(self, batch, batch_idx, stage: str):
        ids, label, mask, token_type_ids = self._prepare_batch(batch)
        output = self(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids)
        label = label.type_as(output)
        loss = F.cross_entropy(output, label)
        acc = (torch.argmax(output, dim=-1) == torch.argmax(label, dim=-1)).float().mean()
        self.log(f"{stage}_loss", loss, on_step=True)
        self.log(f"{stage}_acc", acc, on_step=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        ids, label, mask, token_type_ids = self._prepare_batch(batch)
        output = self(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids)
        return torch.argmax(output, dim=-1)

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, 'test')

    def _prepare_batch(self, batch):
        ids = batch['ids']
        token_type_ids = batch['token_type_ids']
        mask = batch['mask']
        label = batch['target']
        # label = label.unsqueeze(1)
        return ids, label, mask, token_type_ids


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import logging
from logging import WARNING
logging.basicConfig(level=WARNING)

early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-4, patience=2, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    accelerator='gpu',
    gradient_clip_val=0.1,
    # clipping gradients is a hyperparameter and important to prevent divergence
    # of the gradient for recurrent neural networks
    auto_lr_find=True,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
    weights_summary="top",
)

hparams = dict(
    num_classes=df_train.score.unique().size
)

checkpoint = None# "lightning_logs/lightning_logs/version_6/checkpoints/epoch=4-step=5700.ckpt"
if checkpoint is not None:
    model = BERT.load_from_checkpoint(checkpoint, **hparams)
    print(f'Checkpoint {checkpoint} loaded')
else:
    model = BERT(**hparams)


# Disable training of BERT model
for param in model.bert_model.parameters():
    param.requires_grad = False

trainer.validate(model, val_dataloader)


trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


