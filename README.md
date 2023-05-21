

There are three tasks in total

## Task 1

The initial task is to modify the `perturb_method` parameter in the scripts `eval_tk_instruct_english_perturb.sh` and `eval_tk_instruct_xlingual_perturb.sh`. After changing this parameter, execute the scripts to run experiments. You can find the specific perturbation methods in `src/ni_dataset_perturb.py`. To ensure reproducible results, each perturbation method should be run three times with the seed set to 1, 2, and 3, respectively.

## Task 2

The second task is to conduct experiments on `induction_data`. This will involve modifying the run files in the scripts `eval_tk_instruct_english_orignal.sh` and `eval_tk_instruct_xlingual_orignal.sh`. You will need to change these scripts to point to `Tk-Instruct/src/run_s2s_induction.py`, then run them.

## Task 3

We constructed a dataset named Para-Instructions that contains multiple human-oriented instructions for each task.

Para-Instructions are files tmp/new_instruction_*.csv.

The third task is to test Para-Instructions. You'll need to replace the output in `tmp/new_instruction.ipynb` with `data/splits/default/dev_tasks.txt`. Use the suffix for the new definition and no suffix for the original one. Then, execute `scripts/eval_tk_instruct_english_orignal.sh` and conduct an experiment to compare the results between the new and original definitions.



## Citation

```bib
@misc{gu2023robustness,
      title={Robustness of Learning from Task Instructions}, 
      author={Jiasheng Gu and Hongyu Zhao and Hanzi Xu and Liangyu Nie and Hongyuan Mei and Wenpeng Yin},
      year={2023},
      eprint={2212.03813},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



Forked from [GitHub - yizhongw/Tk-Instruct: Tk-Instruct is a Transformer model that is tuned to solve many NLP tasks by following instructions.](https://github.com/yizhongw/Tk-Instruct)
