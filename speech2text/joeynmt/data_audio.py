# coding: utf-8
"""
Data module
"""
import sys
import random
import os
import os.path
from typing import Optional
import logging
import csv

from torchtext.legacy.datasets import TranslationDataset
from torchtext.legacy import data
from torchtext.legacy.data import Dataset, Iterator, Field

from typing import List, Dict, Tuple, Union
from torch import Tensor
import torchaudio

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab_audio, Vocabulary

logger = logging.getLogger(__name__)


def load_audio_data(data_cfg: dict, datasets: list = None)\
        -> (Dataset, Dataset, Optional[Dataset], Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :param datasets: list of dataset names to load
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    if datasets is None:
        datasets = ["train", "dev", "test"]

    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError('Please specify at least one data source path.')

    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.RawField()
    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = None
    if "train" in datasets and train_path is not None:
        logger.info("Loading training data...")
        train_data = AudioDataset(path=train_path,
                                  fields=(src_field, trg_field),
                                  tsv_name="train")

        random_train_subset = data_cfg.get("random_train_subset", -1)
        if random_train_subset > -1:
            # select this many training examples randomly and discard the rest
            keep_ratio = random_train_subset / len(train_data)
            keep, _ = train_data.split(
                split_ratio=[keep_ratio, 1 - keep_ratio],
                random_state=random.getstate())
            train_data = keep

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    assert (train_data is not None) or (src_vocab_file is not None)
    assert (train_data is not None) or (trg_vocab_file is not None)

    logger.info("Building vocabulary...")
    src_vocab = Vocabulary() # TODO: Was genau muss das dann sein?
    trg_vocab = build_vocab_audio(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)

    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("Loading dev data...")
        dev_data = AudioDataset(path=dev_path, 
                                fields=(src_field, trg_field),
                                tsv_name="train")

    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test data...")
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = AudioDataset(path=test_path,
                fields=(src_field, trg_field),
                tsv_name="test")
        else:
            # no target is given -> create dataset from src only
            test_data = MonoAudioDataset(path=test_path, field=src_field,
                                    tsv_name="test" )
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    logger.info("Data loaded.")
    return train_data, dev_data, test_data, src_vocab, trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch

class MonoAudioDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, field, tsv_name: str, **kwargs) -> None:
        examples = []
        
        fields = [('src', field)]

        self._path = os.fspath(path)
        self._tsv = os.path.join(self._path, f"{tsv_name}.tsv")
        self._clips = os.path.join(self._path, f"{tsv_name}_audio")
        
        examples = []

        with open(self._tsv, "r") as tsv_:
            walker = csv.reader(tsv_, delimiter="\t")
            for index, line in enumerate(walker):
                # Note: We simply ignore the line content.
                features = load_and_process_commonvoice_item(self._clips, index)
                examples.append(data.Example.fromlist( [features], fields ))

        super().__init__(examples, fields, **kwargs)

    def __len__(self):
        return len(self.examples)

    

# Partially taken from https://github.com/pytorch/audio/blob/master/torchaudio/datasets/commonvoice.py#L32
# Also based on `torchtext.legacy.datasets.TranslationDataset` to be compatible with JoeyNMT
# This is a modified version.
class AudioDataset(Dataset):
    """Dataset for our audio"""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __len__(self) -> int:
        return len(self.examples)

    def __init__(self, path: str, fields, tsv_name: str, **kwargs) -> None:

        # Note: `src` and `trg` are used by JoeyNMT in vocabulary.py
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        self._path = os.fspath(path)
        self._tsv = os.path.join(self._path, f"{tsv_name}.tsv")
        self._clips = os.path.join(self._path, f"{tsv_name}_audio")
        
        examples = []

        with open(self._tsv, "r") as tsv_:
            walker = csv.reader(tsv_, delimiter="\t")
            for index, line in enumerate(walker):
                features = load_and_process_commonvoice_item(self._clips, index)
                examples.append(data.Example.fromlist( [features, line[0]], fields ))

        super().__init__(examples, fields, **kwargs)

# Taken from https://github.com/pytorch/audio/blob/8a347b62cf5c907d2676bdc983354834e500a282/torchaudio/datasets/commonvoice.py#L12
# This is a modified version.
def load_and_process_commonvoice_item(clips_folder: str, index) -> Tuple[Tensor, int, Dict[str, str]]:

    # TODO: Some processing of data

    filename = os.path.join(clips_folder, f"{index}.mp3")
    waveform, sample_rate = torchaudio.load(filename)

    dic = dict(path=filename)

    return waveform
