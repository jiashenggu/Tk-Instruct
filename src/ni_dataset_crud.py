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
    def __init__(self, *args, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task


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
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(split_dir, "test_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test"
                }),
        ]

    def _generate_examples(self, path=None, task_dir=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
        crud = DataAugmentation()
        with open(path, encoding="utf-8") as split_f:
            for line in split_f:
                task_name = line.strip()
                task_path = os.path.join(task_dir, task_name + ".json")
                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)
                    task_data["Task"] = task_name
                    if "Instruction Source" in task_data:
                        task_data.pop("Instruction Source")
                    Definition = task_data['Definition'][0]

                    # Definition_crud = crud.delete_ratio(Definition, ratio=0.1)
                    # Definition_crud = crud.delete_num(Definition, 5)
                    # Definition_crud = crud.delete_stopwords(Definition)
                    # Definition_crud = crud.insert_mask(Definition, num_mask=10)
                    # Definition_crud = crud.repeat_sentences(Definition, index = None)
                    # Definition_crud = crud.replace_num(Definition, num_mask=10)
                    # Definition_crud = crud.shuffle_sentences(Definition)

                    print("Definition_native: ", Definition)
                    # print("Definition_change: ", Definition_crud)

                    # task_data['Definition'][0] = Definition_crud

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
                        example["Instance"] = instance
                        yield f"{task_name}_{idx}", example

class DataAugmentation:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
        self.en = spacy.load('en_core_web_sm')
        

    def delete_num(self, Definition, num=5):

        splited_definition = Definition.split()
        index = [i for i in range(len(splited_definition))]
        index = random.sample(index, len(splited_definition)-num)
        index.sort()
        Definition_crud = " ".join([splited_definition[i] for i in index])
        return Definition_crud
    def delete_stopwords(self, Definition):

        #loading the english language small model of spacy
        
        stopwords = self.en.Defaults.stop_words
        lst=[]
        for token in Definition.split():
            if token.lower() not in stopwords:    #checking whether the word is not 
                lst.append(token)                    #present in the stopword list.
        Definition_crud = " ".join(lst)     
        return Definition_crud

    def insert_mask(self, Definition, num_mask=5):

        token_list = Definition.split()


        num = 0
        while num < num_mask:
            num += 1
            insert_position = random.randint(1, len(token_list) - 1)
            token_list.insert(insert_position, '[MASK]')
    

        input_txt = ' '.join(token_list)

        inputs = self.tokenizer(input_txt, return_tensors='pt')

        
        input_ids = inputs['input_ids'][0].numpy()
        if input_ids.shape[0]>512:
            return Definition
        outputs = self.model(**inputs)
        predictions = outputs[0]

        _, sorted_idx = predictions[0].sort(dim=-1, descending=True)
        
        for k in range(1):
            predicted_index = [sorted_idx[i, k].item() for i in range(0, len(predictions[0])-1)]
            predicted_token = []
            for x in range(1, len(predictions[0])-1):
                if input_ids[x] == 103:
                    predicted_token.append(self.tokenizer.convert_ids_to_tokens([predicted_index[x]])[0])
        copy_token = predicted_token.copy()
        token_list_copy = token_list.copy()
        for i, token in enumerate(token_list):
            if token == '[MASK]':
                if len(predicted_token)==0:
                    print(Definition)
                    print(token_list_copy)
                    print(copy_token)
                token_list[i] = predicted_token.pop(0)
        final_tokens = []
        for token in token_list:
            if token.startswith('##'):
                final_tokens[-1] = final_tokens[-1] + token[2:]
            else:
                final_tokens.append(token)

        return " ".join(final_tokens)
        
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

    def replace_num(self, Definition, num_mask):

        inputs = self.tokenizer(Definition, return_tensors='pt')
        input_ids = inputs['input_ids'][0]

        index = [i for i in range(len(input_ids))]

        index = random.sample(index, num_mask)

        index.sort()
        for i in index:
            input_ids[i] = 103


        if len(input_ids)>512:
            return Definition
        inputs['input_ids'][0] = input_ids
        outputs = self.model(**inputs)
        predictions = outputs[0]

        _, sorted_idx = predictions[0].sort(dim=-1, descending=True)
        

        for k in range(1):
            predicted_index = [sorted_idx[i, k].item() for i in range(0, len(predictions[0])-1)]
            for x in range(1, len(predictions[0])-1):
                if input_ids[x] == 103:
                    input_ids[x] = predicted_index[x]

        return self.tokenizer.decode(input_ids[1: -1])