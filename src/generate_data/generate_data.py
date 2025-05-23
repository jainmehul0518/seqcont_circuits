import random

import torch
from typing import Optional
import copy

def get_top_preds_moredata(
    prompt: str,
    answer: str,
    model, 
    incor: str,
    prepend_space_to_answer: Optional[bool] = True,
    print_details: Optional[bool] = True,
    prepend_bos: Optional[bool] = True, 
    top_k: Optional[int] = 10,
):
    """
    """
    if prepend_space_to_answer and not answer.startswith(" "):
        answer = " " + answer
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    answer_tokens = model.to_tokens(answer, prepend_bos=False)
    tokens = torch.cat((prompt_tokens, answer_tokens), dim=1)
    prompt_str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)
    answer_str_tokens = model.to_str_tokens(answer, prepend_bos=False)
    prompt_length = len(prompt_str_tokens)
    answer_length = len(answer_str_tokens)
    logits = model(tokens)
    if logits.shape[0] == 1:
        logits = logits.squeeze(0)
    probs = logits.softmax(dim=-1)

    answer_ranks = []

    for index in range(prompt_length, prompt_length + answer_length):
        answer_token = tokens[0, index]
        answer_str_token = answer_str_tokens[index - prompt_length]
        # Offset by 1 because models predict the NEXT token
        token_probs = probs[index - 1]
        sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
        correct_rank = torch.arange(len(sorted_token_values))[
            (sorted_token_values == answer_token).cpu()
        ].item()
        answer_ranks.append((answer_str_token, correct_rank))

    k = top_k
    while k < 500:
        toks = [model.to_string(tok) for tok in sorted_token_values[:k]]
        if incor in toks:
            incor_ind = toks.index(incor)
            break
        else:
            k += 50

    if k < 500:
        return logits[index-1, sorted_token_values[:k]], sorted_token_probs[:k], toks, incor_ind
    else:
        return [], [], [], 'cont'


######
def generate_prompts_list(x, y, words, verb):
    """
    We generate the numwords first as it's the least likely to have prompts that meet our prob bound conds for corr answer vs incorr answer.
    We then use this template for months and numerals by just replacing the numwords seq member and checking if prob bound conds still hold, filtering out those that don't.
    """
    numwords = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
    prompts_list = []
    for j in range(1024): # this must come first else 1 2 3 overrepresented!
        for i in range(x, y):
            rand_words = random.sample(words, k=5)
            prompt_dict = {
                'S1': numwords[i],
                'S2': numwords[i+1],
                'S3': numwords[i+2],
                'S4': numwords[i+3],
                'corr': f" {numwords[i+4]}",
                'incorr': f" {numwords[i+3]}",
                'text': f"{rand_words[0]} was {verb} in {numwords[i]}. {rand_words[1]} was {verb} in {numwords[i+1]}. {rand_words[2]} was {verb} in {numwords[i+2]}. {rand_words[3]} was {verb} in {numwords[i+3]}. {rand_words[4]} was {verb} in",
            }
            prompts_list.append(prompt_dict)
    return prompts_list

###############

def replace_nw_seqtype(data_list, repl_type):
    if repl_type == 'numerals':
        repl_dict = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12'}
    elif repl_type == 'months':
        repl_dict = {'one': 'January', 'two': 'February', 'three': 'March', 'four': 'April', 'five': 'May', 'six': 'June', 'seven': 'July', 'eight': 'August', 'nine': 'September', 'ten': 'October', 'eleven': 'November', 'twelve': 'December'}
    elif repl_type == "numerals_step_2":
        repl_dict = {'one': '2', 'two': '4', 'three': '6', 'four': '8', 'five': '10', 'six': '12', 'seven': '14', 'eight': '16', 'nine': '18', 'ten': '20', 'eleven': '22', 'twelve': '24'}
    elif repl_type == "numerals_step_3":
        repl_dict = {'one': '1', 'two': '4', 'three': '7', 'four': '10', 'five': '13', 'six': '16', 'seven': '19', 'eight': '22', 'nine': '25', 'ten': '28', 'eleven': '31', 'twelve': '34'}
    elif repl_type == "numerals_times_2":
        repl_dict = {'one': '1', 'two': '2', 'three': '4', 'four': '8', 'five': '16', 'six': '32', 'seven': '64', 'eight': '128', 'nine': '256', 'ten': '512', 'eleven': '1024', 'twelve': '2048'}
    elif repl_type == "numerals_alternate":
        repl_dict = {'one': '1', 'two': '2', 'three': '1', 'four': '2', 'five': '1', 'six': '2', 'seven': '1', 'eight': '2', 'nine': '1', 'ten': '2', 'eleven': '1', 'twelve': '2'}
    elif repl_type == "decimal_ascend":
        repl_dict = {'one': '0.0', 'two': '0.5', 'three': '1.0', 'four': '1.5', 'five': '2.0', 'six': '2.5', 'seven': '3.0', 'eight': '3.5', 'nine': '4.0', 'ten': '4.5', 'eleven': '5.0', 'twelve': '5.5'}
    elif repl_type == 'fibonacci':
        repl_dict = {
            'one': '1',   'two': '1',   'three': '2',  'four': '3',
            'five': '5',  'six': '8',   'seven': '13', 'eight': '21',
            'nine': '34','ten': '55','eleven': '89','twelve': '144'
        }
    elif repl_type == 'fibonacci_words':
        repl_dict = {
            'one': 'one',   'two': 'one',    'three': 'two',   'four': 'three',
            'five': 'five', 'six': 'eight',  'seven': 'thirteen',
            'eight': 'twenty one','nine': 'thirty four',
            'ten': 'fifty five','eleven': 'eighty nine',
            'twelve': 'one hundred forty four'
        }
    elif repl_type == 'alternating_sign':
        repl_dict = {'one': '1', 'two': '-2', 'three': '3', 'four': '-4', 'five': '5', 'six': '-6', 'seven': '7', 'eight': '-8', 'nine': '9', 'ten': '-10', 'eleven': '11', 'twelve': '-12'}
    elif repl_type == 'descending_num':
        repl_dict = {'one': '12', 'two': '11', 'three': '10', 'four': '9', 'five': '8', 'six': '7', 'seven': '6', 'eight': '5', 'nine': '4', 'ten': '3', 'eleven': '2', 'twelve': '1'}
    elif repl_type == 'descending_num_words':
        repl_dict = {'one': 'twelve', 'two': 'eleven', 'three': 'ten', 'four': 'nine', 'five': 'eight', 'six': 'seven', 'seven': 'six', 'eight': 'five', 'nine': 'four', 'ten': 'three', 'eleven': 'two', 'twelve': 'one'}
       

    out = copy.deepcopy(data_list)
    for item in out:
        # Replace month names in key-value pairs
        for key in list(item.keys()):  # list() to avoid 'RuntimeError: dictionary changed size during iteration'
            value = item[key]
            if value in repl_dict:
                item[key] = repl_dict[value]
            elif key == 'corr' or key == 'incorr':
                item[key] = " " + repl_dict[value.replace(" ", '')]

        # Replace month names in text fields
        if 'text' in item:
            text = item['text']
            for month_name, month_num in repl_dict.items():
                text = text.replace(month_name, str(month_num))
            item['text'] = text

    return out

###############
def filter_to_single_token(model, words):
    return [w for w in words if len(model.tokenizer.tokenize(w)) == 1]

###############
def get_good_prompts_nw_months(model, prompts_list, prompts_list_months):
    all_probs = []
    all_probs_m = []
    good_prompts = []
    good_prompts_months = []

    eight_group = []
    eight_group_months = []
    for prompt_dict, prompt_dict_months in zip(prompts_list, prompts_list_months):
        prompt = prompt_dict['text']
        answer = prompt_dict['corr']
        incor = prompt_dict['incorr']
        prompt_months = prompt_dict_months['text']
        answer_months = prompt_dict_months['corr']
        incor_months = prompt_dict_months['incorr']

        logs, probs, toks, incor_ind = get_top_preds_moredata(
            prompt = prompt,
            answer = answer,
            model = model,
            incor = incor
        )

        logs_months, probs_months, tok_months, incor_ind_months = get_top_preds_moredata(
            prompt = prompt_months,
            answer = answer_months,
            model = model,
            incor = incor_months
        )

        if toks[0] == answer and probs[0] > 2*probs[toks.index(incor)]:
            if tok_months[0] == answer_months and probs_months[0] > 2*probs_months[tok_months.index(incor_months)]:
                eight_group.append(prompt_dict)
                eight_group_months.append(prompt_dict_months)
                all_probs.append(probs)
                all_probs_m.append(probs_months)
                if prompt_dict['S1'] == 'eight':
                    if len(eight_group) == 8:
                        good_prompts += eight_group
                        good_prompts_months += eight_group_months
                        print(len(good_prompts))
                    eight_group = []
                    eight_group_months = []
                if len(good_prompts) == 1024:
                    break
    return good_prompts, good_prompts_months, all_probs

###############
def get_good_prompts_numerals(model, prompts_list):
    logit_diffs = []
    all_probs = []
    good_prompts = []

    for prompt_dict in prompts_list:
        prompt = prompt_dict['text']
        answer = prompt_dict['corr']
        incor = prompt_dict['incorr']

        logs, probs, toks, incor_ind = get_top_preds_moredata(
            prompt = prompt,
            answer = answer,
            model = model,
            incor = incor
        )

        ### debugging ###
        print("answer=\n")
        print(answer)

        print("\nincor=\n")
        print(incor)

        print("\ntoks = \n")
        print(toks)

        print("\nincor_ind =\n")
        print(incor_ind)

        print("\nprobs = \n")
        print(probs)
        ### debugging ###

        if incor_ind == 'cont':
            continue

        if toks[0] == answer and probs[0] > 2*probs[toks.index(incor)]:
            all_probs.append(probs)
            l_diff = logs[0] - logs[incor_ind]
            logit_diffs.append(l_diff.item())
            good_prompts.append(prompt_dict)
            if len(good_prompts) == 1024:
                break
    return good_prompts, all_probs

###############
def generate_prompts_list_corr(prompt_list):
    outlist = []
    for prompt_dict in prompt_list:
        r1 = random.randint(1, 12)
        r2 = random.randint(1, 12)
        while True:
            r3 = random.randint(1, 12)
            r4 = random.randint(1, 12)
            if r4 - 1 != r3:
                break
        # new_text = prompt_dict['text'].replace(prompt_dict['S1'], str(r1)).replace(prompt_dict['S2'], str(r2)).replace(prompt_dict['S3'], str(r3)).replace(prompt_dict['S4'], str(r4))
        text_list = prompt_dict['text'].split(".")
        new_text_list = []
        new_text_list.append(text_list[0].replace(prompt_dict['S1'], str(r1)))
        new_text_list.append(text_list[1].replace(prompt_dict['S2'], str(r2)))
        new_text_list.append(text_list[2].replace(prompt_dict['S3'], str(r3)))
        new_text_list.append(text_list[3].replace(prompt_dict['S4'], str(r4)))
        new_text_list.append(text_list[4])
        new_text = ".".join(new_text_list)
        
        new_prompt_dict = {
            'S1': str(r1),
            'S2': str(r2),
            'S3': str(r3),
            'S4': str(r4),
            'corr': prompt_dict['corr'],
            'incorr': prompt_dict['incorr'],
            'text': new_text
        }
        outlist.append(new_prompt_dict)

    return outlist

# prompts_list_2 = generate_prompts_list_corr(prompts_list)

# import pickle
# from google.colab import files

# with open('randDS_numerals.pkl', 'wb') as file:
#     pickle.dump(prompts_list_2, file)
# files.download('randDS_numerals.pkl')