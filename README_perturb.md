There are three tasks in total
The first task is to interfere with the task definition by changing the perturb_method parameter in scripts/eval_tk_instruct_english_perturb.sh and scripts/eval_tk_instruct_xlingual_perturb.sh and running it to do1 experiments. The specific perturbation method is in src/ni_dataset_perturb.py. Each perturbation method is run three times to ensure reproducibility, setting the seed to 1, 2, and 3 respectively.

The second task is to experiment on induction_data by changing the run files in scripts/eval_tk_instruct_english_orignal.sh and scripts/eval_tk_instruct_xlingual_orignal.sh to Tk-Instruct/src/run_s2s_induction.py and run it.

The third task is to experiment on the newly written definition by replacing the output in tmp/new_instruction.ipynb with data/splits/default/dev_tasks.txt, with the suffix for the newly written one and no suffix for the original one, and running scripts/eval_tk_instruct_ english_orignal.sh and do an experiment for comparison respectively.
