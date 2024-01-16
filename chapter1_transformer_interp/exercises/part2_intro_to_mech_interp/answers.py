# %%

import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part2_intro_to_mech_interp', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "mps")

MAIN = __name__ == "__main__"

# %% 1️⃣ TRANSFORMERLENS: INTRODUCTION


if MAIN:
	gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

# %%


if MAIN:
	model_description_text = '''## Loading Models
	
	HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 
	
	For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''
	
	loss = gpt2_small(model_description_text, return_type="loss")
	print("Model loss:", loss)

# %%


if MAIN:
	print(gpt2_small.to_str_tokens("gpt2"))
	print(gpt2_small.to_tokens("gpt2"))
	print(gpt2_small.to_string([50256, 70, 457, 17]))

# %%


if MAIN:
	logits: Tensor = gpt2_small(model_description_text, return_type="logits")
	prediction = logits.argmax(dim=-1).squeeze()[:-1]
	# FLAT SOLUTION
	# YOUR CODE HERE - get the model's prediction on the text
	true_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
	is_correct = (prediction == true_tokens)
	
	print(f"Model accuracy: {is_correct.sum()}/{len(true_tokens)}")
	print(f"Correct words: {gpt2_small.to_str_tokens(prediction[is_correct])}")

# %%


if MAIN:
	gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
	gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
	gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

# %%


if MAIN:
	attn_patterns_layer_0 = gpt2_cache["pattern", 0]

# %%


if MAIN:
	attn_patterns_layer_0_copy = gpt2_cache["blocks.0.attn.hook_pattern"]
	
	t.testing.assert_close(attn_patterns_layer_0, attn_patterns_layer_0_copy)

# %%


if MAIN:
	layer0_pattern_from_cache = gpt2_cache["pattern", 0]
	
	# FLAT SOLUTION
	# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the steps of the attention calculation (dot product, masking, scaling, softmax)
	q, k = gpt2_cache["q", 0], gpt2_cache["k", 0]
	seq, nhead, headsize = q.shape
	layer0_attn_scores = einops.einsum(q, k, "seqQ n h, seqK n h -> n seqQ seqK")
	mask = t.triu(t.ones((seq, seq), dtype=bool), diagonal=1).to(device)
	layer0_attn_scores.masked_fill_(mask, -1e9)
	layer0_pattern_from_q_and_k = (layer0_attn_scores / headsize**0.5).softmax(-1)
	
	t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
	print("Tests passed!")

# %%


if MAIN:
	print(type(gpt2_cache))
	attention_pattern = gpt2_cache["pattern", 0, "attn"]
	print(attention_pattern.shape)
	gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)
	
	print("Layer 0 Head Attention Patterns:")
	display(cv.attention.attention_patterns(
		tokens=gpt2_str_tokens, 
		attention=attention_pattern,
		attention_head_names=[f"L0H{i}" for i in range(12)],
	))

# %% 2️⃣ FINDING INDUCTION HEADS


if MAIN:
	cfg = HookedTransformerConfig(
		d_model=768,
		d_head=64,
		n_heads=12,
		n_layers=2,
		n_ctx=2048,
		d_vocab=50278,
		attention_dir="causal",
		attn_only=True, # defaults to False
		tokenizer_name="EleutherAI/gpt-neox-20b", 
		seed=398,
		use_attn_result=True,
		normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
		positional_embedding_type="shortformer"
	)

# %%


if MAIN:
	weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()
	
	if not weights_dir.exists():
		url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
		output = str(weights_dir)
		gdown.download(url, output)

# %%


if MAIN:
	model = HookedTransformer(cfg)
	pretrained_weights = t.load(weights_dir, map_location=device)
	model.load_state_dict(pretrained_weights)

# %%


if MAIN:
	text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
	
	logits, cache = model.run_with_cache(text, remove_batch_dim=True)

# %%


if MAIN:
	# FLAT SOLUTION
	# YOUR CODE HERE - visualize attention
	str_tokens = model.to_str_tokens(text)
	for layer in range(model.cfg.n_layers):
		attention_pattern = cache["pattern", layer]
		display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))

# %%

def current_attn_detector(cache: ActivationCache) -> List[str]:
	'''
	Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
	'''
	attn_heads = []
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			attention_pattern = cache["pattern", layer][head]
			# take avg of diagonal elements
			score = attention_pattern.diagonal().mean()
			if score > 0.4:
				attn_heads.append(f"{layer}.{head}")
	return attn_heads

def prev_attn_detector(cache: ActivationCache) -> List[str]:
	'''
	Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
	'''
	attn_heads = []
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			attention_pattern = cache["pattern", layer][head]
			# take avg of sub-diagonal elements
			score = attention_pattern.diagonal(-1).mean()
			if score > 0.4:
				attn_heads.append(f"{layer}.{head}")
	return attn_heads

def first_attn_detector(cache: ActivationCache) -> List[str]:
	'''
	Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
	'''
	attn_heads = []
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			attention_pattern = cache["pattern", layer][head]
			# take avg of 0th elements
			score = attention_pattern[:, 0].mean()
			if score > 0.4:
				attn_heads.append(f"{layer}.{head}")
	return attn_heads



if MAIN:
	print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
	print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
	print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))

# %%

def generate_repeated_tokens(
	model: HookedTransformer,
	seq_len: int,
	batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
	'''
	Generates a sequence of repeated random tokens

	Outputs are:
		rep_tokens: [batch, 1+2*seq_len]
	'''
	prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
	rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
	rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
	return rep_tokens



def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
	'''
	Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

	Should use the `generate_repeated_tokens` function above

	Outputs are:
		rep_tokens: [batch, 1+2*seq_len]
		rep_logits: [batch, 1+2*seq_len, d_vocab]
		rep_cache: The cache of the model run on rep_tokens
	'''
	rep_tokens = generate_repeated_tokens(model, seq_len, batch)
	rep_logits, rep_cache = model.run_with_cache(rep_tokens)
	return rep_tokens, rep_logits, rep_cache



if MAIN:
	seq_len = 50
	batch = 1
	(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
	rep_cache.remove_batch_dim()
	rep_str = model.to_str_tokens(rep_tokens)
	model.reset_hooks()
	log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()
	
	print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
	print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")
	
	plot_loss_difference(log_probs, rep_str, seq_len)

# %%

# YOUR CODE HERE - display the attention patterns stored in `rep_cache`, for each layer

# %%

def induction_attn_detector(cache: ActivationCache) -> List[str]:
	'''
	Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

	Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
	'''
	attn_heads = []
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			attention_pattern = cache["pattern", layer][head]
			# take avg of (-seq_len+1)-offset elements
			seq_len = (attention_pattern.shape[-1] - 1) // 2
			score = attention_pattern.diagonal(-seq_len+1).mean()
			if score > 0.4:
				attn_heads.append(f"{layer}.{head}")
	return attn_heads



if MAIN:
	print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%
def hook_function(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint
) -> Float[Tensor, "batch heads seqQ seqK"]:

    # modify attn_pattern (can be inplace)
    return attn_pattern

seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    # Get an average score per head
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    # Store the result.
    induction_score_store[hook.layer(), :] = induction_score


pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=400
)
# %%
seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(gpt2_small, seq_len, batch)

induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device)

gpt2_small.run_with_hooks(rep_tokens_10, return_type=None, fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)])

imshow(
    induction_score_store, 
    labels={"x": "Head", "y": "Layer"}, 
    title="Induction Score by Head", 
    text_auto=".2f",
    width=900, height=400
)

# %%
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    if hook.layer() in [5, 6, 7]:
        print("Layer: ", hook.layer())
        display(
			cv.attention.attention_patterns(
				tokens=gpt2_small.to_str_tokens(rep_tokens[0]), 
				attention=pattern.mean(0)
				)
			)


gpt2_small.run_with_hooks(rep_tokens_10, fwd_hooks=[(pattern_hook_names_filter, visualize_pattern_hook)])

# %%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]]
    pass


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")