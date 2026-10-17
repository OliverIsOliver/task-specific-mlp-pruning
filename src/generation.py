import json
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .evaluation import correct
from .storage import write_json


# Preserve the stopping rules used to produce the saved experiment responses.
class Finished(StoppingCriteria):
    def __init__(self, tokenizer, start, task):
        self.tokenizer, self.start, self.task = tokenizer, start, task

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.start:], skip_special_tokens=True)
        stop = '\n' in text.lstrip('\r\n ') if self.task == 'arithmetic' else '}' in text
        return torch.tensor([stop], device=input_ids.device)


def synchronize(device):
    if device=='mps':
        torch.mps.synchronize()
    elif device=='cuda':
        torch.cuda.synchronize()


@torch.inference_mode()
def generate_food(row,tokenizer,model,device,grade=True,max_tokens=None,stop=None):
    ids = tokenizer(row['prompt'],return_tensors='pt').input_ids.to(device)
    tokens, past, finished = [],None,False
    limit = max_tokens if max_tokens is not None else 128 if row['task']=='food' else 16
    synchronize(device)
    began = time.perf_counter()
    # Greedy decoding picks the highest-logit token, reusing the attention KV cache.
    for _ in range(limit):
        output = model(input_ids=ids,past_key_values=past,use_cache=True,logits_to_keep=1)
        token = int(output.logits[0,-1].argmax())
        tokens.append(token)
        response = tokenizer.decode(tokens,skip_special_tokens=True)
        finished = token==tokenizer.eos_token_id or (stop in response if stop is not None
                    else ']' in response if row['task']=='food' else '\n' in response.lstrip('\r\n '))
        if finished:
            break
        past = output.past_key_values
        ids = torch.tensor([[token]],device=device)
    synchronize(device)
    result = dict(id=row['id'],task=row['task'],suite=row['suite'],response=response,
                  generated_tokens=len(tokens),truncated=not finished,seconds=time.perf_counter()-began)
    if grade:
        result.update(expected=row['answer'],exact_match=correct(row['task'], row['answer'], response))
    return result


# Appointment and policy answers stop when their JSON object closes.
def generate_json(rows, tokenizer, model, device, path, cached=()):
    runs = json.loads(path.read_text()) if path.exists() else list(cached)
    done = {r['id'] for r in runs}
    for i, row in enumerate(rows):
        if row['id'] not in done:
            ids = tokenizer(row['prompt'], return_tensors='pt').input_ids.to(device)
            start = ids.shape[1]
            with torch.inference_mode():
                tokens = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    do_sample=False, num_beams=1, max_new_tokens=64, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([Finished(tokenizer, start, row['task'])]))
            response = tokenizer.decode(tokens[0, start:], skip_special_tokens=True)
            runs.append(dict(id=row['id'], task=row['task'], response=response,
                generated_tokens=tokens.shape[1]-start,
                truncated=int(tokens[0,-1])!=tokenizer.eos_token_id and '}' not in response))
            write_json(path, runs)
        if (i+1) % 20 == 0:
            print(f'{rows[0]["task"]} {path.stem}: {i+1}/{len(rows)}', flush=True)
    assert len(runs) == len({r['id'] for r in runs}) == len(rows)
    return runs


# Arithmetic answers stop at the first newline after the integer.
def generate_addition(rows, tokenizer, model, device, path):
    runs = json.loads(path.read_text()) if path.exists() else []
    done = {r['id'] for r in runs}
    for i,row in enumerate(rows):
        if row['id'] not in done:
            ids = tokenizer(row['prompt'],return_tensors='pt').input_ids.to(device)
            start = ids.shape[1]
            with torch.inference_mode():
                tokens = model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                    num_beams=1,max_new_tokens=64,use_cache=True,pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([Finished(tokenizer,start,'arithmetic')]))
            response = tokenizer.decode(tokens[0,start:],skip_special_tokens=True)
            stopped = int(tokens[0,-1])==tokenizer.eos_token_id or '\n' in response.lstrip('\r\n ')
            runs.append(dict(id=row['id'],family=row['family'],response=response,
                generated_tokens=tokens.shape[1]-start,truncated=not stopped))
            write_json(path,runs)
        if (i+1)%20==0:
            print(f'Addition {path.stem}: {i+1}/{len(rows)}',flush=True)
    assert len(runs)==len({r['id'] for r in runs})==len(rows)
    return runs
