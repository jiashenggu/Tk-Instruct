# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Natural Instruction V2 Dataset."""


import json
import os
from posixpath import split
import random
import datasets
import spacy
import random
import transformers
from transformers import BertTokenizer, BertModel, BertForMaskedLM

transformers.logging.set_verbosity_error()

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
"""

_DESCRIPTION = """
Natural-Instructions v2 is a benchmark of 1,600+ diverse language tasks and their expert-written instructions. 
It covers 70+ distinct task types, such as tagging, in-filling, and rewriting. 
These tasks are collected with contributions of NLP practitioners in the community and 
through an iterative peer review process to ensure their quality. 
"""

_URL = "https://instructions.apps.allenai.org/"

class NIConfig(datasets.BuilderConfig):
    def __init__(self, *args, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, perturb_method=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task
        self.perturb_method = perturb_method


class NaturalInstructions(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = NIConfig
    BUILDER_CONFIGS = [
        NIConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Contributors": datasets.Value("string"),
                    "Source": [datasets.Value("string")],
                    "URL": [datasets.Value("string")],
                    "Categories": [datasets.Value("string")],
                    "Reasoning": [datasets.Value("string")],
                    "Definition": [datasets.Value("string")],
                    "Positive Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Negative Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Input_language": [datasets.Value("string")],
                    "Output_language": [datasets.Value("string")],
                    "Instruction_language": [datasets.Value("string")],
                    "Domains": [datasets.Value("string")],
                    # "Instances": [{
                    #     "input": datasets.Value("string"),
                    #     "output": [datasets.Value("string")]
                    # }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        "output": [datasets.Value("string")]
                    },
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/allenai/natural-instructions",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            dl_path = dl_manager.download_and_extract(_URL)
            self.config.data_dir = self.config.data_dir or os.path.join(dl_path, "splits")
            self.config.task_dir = self.config.task_dir or os.path.join(dl_path, "tasks")

        split_dir = self.config.data_dir
        task_dir = self.config.task_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(split_dir, "train_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(split_dir, "dev_tasks.txt"), 
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev",
                    "perturb_method": self.config.perturb_method
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(split_dir, "test_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test",
                }),
        ]

    def _generate_examples(self, path=None, task_dir=None, max_num_instances_per_task=None, subset=None, perturb_method=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
        perturb = DataAugmentation()
        # with open(path, encoding="utf-8") as split_f:
        split_f = open(path, encoding="utf-8").readlines()
        task_ids_other = list(range(0, len(split_f)))
        random.shuffle(task_ids_other)
        for task_id, line in enumerate(split_f):
            task_name = line.strip()
            task_path = os.path.join(task_dir, task_name + ".json")
            task_name_other = split_f[task_ids_other[task_id]].strip()
            task_path_other = os.path.join(task_dir, task_name_other + ".json")
            with open(task_path, encoding="utf-8") as task_f:
                s = task_f.read()
                task_data = json.loads(s)
                task_data["Task"] = task_name
                if "Instruction Source" in task_data:
                    task_data.pop("Instruction Source")
                f_other = open(task_path_other, encoding="utf-8").read()
                task_data_other = json.loads(f_other)

                Definition = task_data["Definition"][0]
                

                if perturb_method == "delete_stopwords":
                    Definition_perturb = perturb.delete_stopwords(Definition)
                elif perturb_method == "delete_words":
                    Definition_perturb = perturb.delete_words(Definition, 10)
                elif perturb_method == "insert_words":
                    Definition_perturb = perturb.insert_words(Definition, 10)
                elif perturb_method == "replace_words":
                    Definition_perturb = perturb.replace_words(Definition, 10)
                elif perturb_method == "shuffle_words":
                    Definition_perturb = perturb.shuffle_words(Definition)
                elif perturb_method == "repeat_sentences":
                    Definition_perturb = perturb.repeat_sentences(Definition)
                elif perturb_method == "shuffle_sentences":
                    Definition_perturb = perturb.shuffle_sentences(Definition)
                elif perturb_method == "shuffle_instructions":
                    Definition_perturb = task_data_other['Definition'][0]
                else:
                    Definition_perturb = Definition
                


                print("Definition_native: ", Definition)
                print("Definition_change: ", Definition_perturb)

                task_data['Definition'][0] = Definition_perturb

                all_instances = task_data.pop("Instances")
                if subset == "test":
                    # for testing tasks, 100 instances are selected for efficient evaluation and they are label-balanced.
                    # we put them in the first for reproducibility.
                    # so, we use them here
                    instances = all_instances[:100]
                else:
                    instances = all_instances
                if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                    random.shuffle(instances)
                    instances = instances[:max_num_instances_per_task]
                for idx, instance in enumerate(instances):
                    example = task_data.copy()
                    example["id"] = instance["id"]
                    example["Instance"] = instance
                    yield f"{task_name}_{idx}", example

class DataAugmentation:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        self.en = spacy.load('en_core_web_sm')
        
    @staticmethod
    def connect_token_segments(tokens):
        connected_tokens = []
        for token in tokens:
            if token.startswith("##"):
                connected_tokens[-1] = connected_tokens[-1] + token[2:]
            else:
                connected_tokens.append(token)
        return connected_tokens


    def delete_words(self, Definition, num=5):

        tokens = self.tokenizer.tokenize(Definition)
        tokens = self.connect_token_segments(tokens)

        index = [i for i in range(len(tokens))]

        deleted_index = random.sample(index, num)
        deleted_index = set(deleted_index)

        deleted_tokens = [tokens[i] for i in index if i not in deleted_index]
        Definition_perturb = self.tokenizer.convert_tokens_to_string(deleted_tokens)
        return Definition_perturb
        
    def delete_stopwords(self, Definition):
        
        tokens = self.tokenizer.tokenize(Definition)
        tokens = self.connect_token_segments(tokens)

        stopwords = self.en.Defaults.stop_words
        deleted_tokens=[]
        for token in tokens:
            if token.lower() not in stopwords:
                deleted_tokens.append(token)
        Definition_perturb =  self.tokenizer.convert_tokens_to_string(deleted_tokens)
        return Definition_perturb

    def insert_words(self, Definition, num_mask=5):

        tokens = self.tokenizer.tokenize(Definition)
        if len(tokens)>512:
            return Definition
        tokens = self.connect_token_segments(tokens)

        index = [i for i in range(len(tokens))]

        index = random.sample(index, num_mask)

        for i in index:
            tokens.insert(i, '[MASK]')

        
        
        Definition = self.tokenizer.convert_tokens_to_string(tokens)
        inputs = self.tokenizer(Definition, return_tensors='pt')
        input_ids = inputs['input_ids'][0]
        outputs = self.model(**inputs)
        predictions = outputs[0]

        _, sorted_idx = predictions[0].sort(dim=-1, descending=True)

        predicted_index = [sorted_idx[i, 0].item() for i in range(0, len(predictions[0])-1)]
        for x in range(1, len(predictions[0])-1):
            if input_ids[x] == 103:
                input_ids[x] = predicted_index[x]

        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def replace_words(self, Definition, num_mask=5):

        tokens = self.tokenizer.tokenize(Definition)
        if len(tokens)>512:
            return Definition
        tokens = self.connect_token_segments(tokens)

        index = [i for i in range(len(tokens))]

        index = random.sample(index, num_mask)

        for i in index:
            tokens[i] = '[MASK]'

        if len(tokens)>512:
            return Definition
        
        Definition = self.tokenizer.convert_tokens_to_string(tokens)
        inputs = self.tokenizer(Definition, return_tensors='pt')
        input_ids = inputs['input_ids'][0]
        outputs = self.model(**inputs)
        predictions = outputs[0]

        _, sorted_idx = predictions[0].sort(dim=-1, descending=True)

        predicted_index = [sorted_idx[i, 0].item() for i in range(0, len(predictions[0])-1)]
        for x in range(1, len(predictions[0])-1):
            if input_ids[x] == 103:
                input_ids[x] = predicted_index[x]

        return self.tokenizer.decode(input_ids, skip_special_tokens=True)
    
    def shuffle_words(self, Definition):

        tokens = self.tokenizer.tokenize(Definition)
        tokens = self.connect_token_segments(tokens)

        random.shuffle(tokens)
        return self.tokenizer.convert_tokens_to_string(tokens)
    
    def shuffle_sentences(self, Definition):

        doc = self.en(Definition)
        sents = list(map(str, doc.sents))
        random.shuffle(sents)
        return " ".join(sents)

    def repeat_sentences(self, Definition, index = None):
        doc = self.en(Definition)
        sents = list(map(str, doc.sents))
        if None == index:
            index = random.randint(0, len(sents)-1)
        sents = sents[:index] + [sents[index]] + sents[index:]
        return " ".join(sents)