How to Create and Experiment With a New Dataset!

Edit generate_data.py (seqcont_circuits/src/generate_data)
Choose a dataset name (ex: numerals, months, numerals_step_2)
In replace_nw_seqtype, add an elif statement for repl_type, checking if it matches the dataset name you chose (ex: repl_type == numerals_step_2)
Add a dictionary mapping based on the sequence you want to detect
Create directory: seqcont_circuits/data/{your dataset name}
Run generate_data.ipynb (seqcont_circuits/notebooks/gpt2_expms)
Follow the TODOs in the notebook
Edit run_node_ablation_batched.py (seqcont_circuits/src/iter_node_pruning/run_node_ablation_batched.py)
Add your dataset name to the choices list in parser.add_argument(--task…)
Run run_node_ablation_batched.py with the correct task
The num_samps argument should be the length of one of your prompt types
Ex: the number of prompts in numerals_step_2_prompts_done.pkl is 448 so I set this as num_samps
Edit run_attn_pats.py (seqcont_circuits/src/attn_pats)
Add your dataset name to the choices list in parser.add_argument(--task…)
Edit viz_attn_pat.py (seqcont_circuits/src/attn_pats)
In the task section (lines 57 onward), add an elif statement
elif task == {your dataset name}:
Set disp_toks using a similar format as the above elif statements
Run vis_attn_pat.py with the correct task
