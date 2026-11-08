"""Model loading and batched generation for Qwen3.5 small models on Apple Silicon."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"

# Recommended "thinking mode, text tasks" settings from the Qwen3.5-0.8B model card.
THINKING_SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)
GREEDY = dict(temperature=0.0, presence_penalty=0.0)


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(repo=DEFAULT_MODEL, patch=None):
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(device()).eval()
    if patch:
        from wsp.edit import apply_patch
        apply_patch(model, patch)
    return model, tok


class PresencePenalty(LogitsProcessor):
    """OpenAI-style presence penalty: flat subtraction for every token already generated."""

    def __init__(self, penalty, prompt_len):
        self.penalty, self.prompt_len = penalty, prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        if gen.shape[1] == 0 or self.penalty == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
        return scores - self.penalty * seen.to(scores.dtype)


def render(tok, messages, thinking, tools=None):
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def eos_ids(model, tok):
    """End-of-turn tokens. The repo ships no generation_config.json, so the config default
    only stops on <|endoftext|>; chat turns actually end with <|im_end|>."""
    e = model.generation_config.eos_token_id
    ids = set(e if isinstance(e, list) else [e])
    ids.add(tok.convert_tokens_to_ids("<|im_end|>"))
    return ids


@torch.inference_mode()
def generate(model, tok, conversations, thinking=True, max_new_tokens=4096,
             batch_size=8, sampling=THINKING_SAMPLING, seed=0, on_batch=None, tools=None):
    """Generate for a list of chat conversations. Returns dicts with text, n_tokens, finished."""
    torch.manual_seed(seed)
    stops = eos_ids(model, tok)
    results = []
    for i in range(0, len(conversations), batch_size):
        chunk = conversations[i:i + batch_size]
        texts = [render(tok, c, thinking, tools) for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        plen = enc["input_ids"].shape[1]
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=sorted(stops))
        if sampling.get("temperature", 0) > 0:
            kw.update(do_sample=True, temperature=sampling["temperature"],
                      top_p=sampling.get("top_p", 1.0), top_k=sampling.get("top_k", 0))
        else:
            kw.update(do_sample=False)
        if sampling.get("presence_penalty"):
            kw["logits_processor"] = LogitsProcessorList([PresencePenalty(sampling["presence_penalty"], plen)])
        out = model.generate(**enc, **kw)
        for row in out[:, plen:].tolist():
            n = len(row)
            for j, t in enumerate(row):
                if t in stops:
                    n = j
                    break
            finished = n < len(row) or (len(row) > 0 and row[-1] in stops)
            results.append(dict(text=tok.decode(row[:n], skip_special_tokens=False),
                                n_tokens=n, finished=finished))
        if on_batch:
            on_batch(i + len(chunk), len(conversations))
    return results
