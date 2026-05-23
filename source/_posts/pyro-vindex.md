---
title: Using Pyro VIndex for Batched Latent-State Models
date: 2026-05-23 00:00:00
tags:
  - Pyro
  - probabilistic programming
  - tensor indexing
  - single-cell RNA-seq
---

I ran into `Vindex` through a very ordinary problem: an HMM-style Pyro model where a latent state has to choose one row from a transition matrix. The indexing line looked harmless enough:

```python
next_logits = transition[..., state, :]
```

That can work for a small test case. The trouble usually appears later, after adding a plate or turning on parallel enumeration. At that point `state` is no longer just a scalar-looking value. It may be a tensor carrying Pyro's inference dimensions, and PyTorch treats tensor indices as advanced indices. The result can be perfectly legal PyTorch while still being the wrong shape for the model.

`Vindex` is Pyro's way of making that intent explicit:

```python
from pyro.ops.indexing import Vindex

next_logits = Vindex(transition)[..., state, :]
```

I read that line as: keep the batch dimensions on the left, use `state` to pick the current-state row, and leave the next-state axis alone. It is a small change in code, but a useful change in what the code says.

## Where PyTorch Indexing Gets Awkward

PyTorch indexing is not wrong here. It is just solving a lower-level array problem. Slices keep dimensions, integer indices remove dimensions, and tensor indices broadcast together as advanced indices. That is exactly what you want in many tensor programs. In Pyro, though, some tensor dimensions are not ordinary data axes; they are enumeration or plate axes used by inference. When those axes move, the model can become hard to debug.

| Case | Plain PyTorch-looking code | What can go wrong | `Vindex` version |
| --- | --- | --- | --- |
| Pick transition rows by latent state | `transition[..., state, :]` | An enumerated `state` is a tensor index, so PyTorch may move or combine indexed dimensions in a way that no longer matches Pyro's batch layout. | `Vindex(transition)[..., state, :]` |
| Pick emission parameters by latent state | `emission[..., state, :]` | A scalar test case works, but parallel enumeration adds left-side dimensions that need to be preserved. | `Vindex(emission)[..., state, :]` |
| Use two latent indices | `p[..., x, y, :]` | PyTorch broadcasts `x` and `y` as advanced indices. In a model, they may correspond to separate latent choices. | `Vindex(p)[..., x, y, :]` |
| Mix plate indices and latent indices | `param[plate_id, state]` | The plate axis and the enumeration axis often need manual reshaping to avoid colliding. | `Vindex(param)[plate_id[..., None], state]` |
| Debug a shape mismatch | Add `unsqueeze`, `expand`, `arange`, and `gather` by hand | This is possible, but the indexing code stops looking like the model. | Keep the model indexing in one line with `Vindex` |

The short version: PyTorch indexing describes how arrays are indexed. `Vindex` describes how a Pyro model wants latent choices to index batched parameter tensors.

## The Same Operation in Plain PyTorch

It is useful to know what `Vindex` is saving us from. Suppose the transition tensor has shape:

```text
transition: batch_shape + (num_states, num_states)
state:      batch_shape
```

Then this:

```python
next_logits = Vindex(transition)[..., state, :]
```

is equivalent to a `gather` along the source-state axis:

```python
import torch


def select_transition_rows(transition, state):
    # transition: batch_shape + (K, K)
    # state:      batch_shape, dtype=torch.long
    # returns:    batch_shape + (K,)
    num_next_states = transition.size(-1)
    index = state[..., None, None].expand(*state.shape, 1, num_next_states)
    return torch.gather(transition, dim=-2, index=index).squeeze(-2)
```

For the simple time-indexed example where `transition[t - 1]` is just a `(K, K)` matrix and `state` may have enumeration dimensions, the equivalent can be written with `index_select`:

```python
matrix = transition[t - 1]  # (K, K)
flat_state = state.reshape(-1)
logits = matrix.index_select(0, flat_state)
logits = logits.reshape(state.shape + (matrix.size(-1),))
```

And for an emission tensor with shape `batch_shape + (num_states, num_genes)`, the gather version is the same pattern:

```python
def select_emission_rows(emission, state):
    # emission: batch_shape + (K, G)
    # state:    batch_shape
    # returns:  batch_shape + (G,)
    num_genes = emission.size(-1)
    index = state[..., None, None].expand(*state.shape, 1, num_genes)
    return torch.gather(emission, dim=-2, index=index).squeeze(-2)
```

Those snippets are not hard, but they are also not the line I want to maintain inside a probabilistic model. `Vindex` keeps the row-selection intent visible and leaves Pyro to manage the batch and enumeration dimensions.

## A Concrete Numeric Example

Here is the smallest version of the problem. Say we have three hidden states, so each transition row gives the logits or probabilities for the next state:

```python
import torch
from pyro.ops.indexing import Vindex

transition = torch.tensor([
    [0.10, 0.70, 0.20],  # from state 0
    [0.30, 0.30, 0.40],  # from state 1
    [0.80, 0.10, 0.10],  # from state 2
])

state = torch.tensor(2)
```

If the current hidden state is `2`, both plain indexing and `Vindex` select row 2:

```python
transition[state, :]
# tensor([0.8000, 0.1000, 0.1000])

Vindex(transition)[state, :]
# tensor([0.8000, 0.1000, 0.1000])
```

Now make it batched. Suppose two cells, two sequences, or two independent examples currently sit in states 0 and 2:

```python
transition = torch.tensor([
    [
        [0.10, 0.70, 0.20],
        [0.30, 0.30, 0.40],
        [0.80, 0.10, 0.10],
    ],
    [
        [0.60, 0.20, 0.20],
        [0.10, 0.80, 0.10],
        [0.25, 0.25, 0.50],
    ],
])
# shape: (batch=2, current_state=3, next_state=3)

state = torch.tensor([0, 2])
# shape: (batch=2,)
```

The intended answer is:

```text
batch 0, state 0 -> [0.10, 0.70, 0.20]
batch 1, state 2 -> [0.25, 0.25, 0.50]
```

With `Vindex`, that intent is direct:

```python
Vindex(transition)[..., state, :]
# tensor([[0.1000, 0.7000, 0.2000],
#         [0.2500, 0.2500, 0.5000]])
```

The equivalent plain PyTorch operation is:

```python
index = state[:, None, None].expand(2, 1, 3)
torch.gather(transition, dim=1, index=index).squeeze(1)
# tensor([[0.1000, 0.7000, 0.2000],
#         [0.2500, 0.2500, 0.5000]])
```

This is the main troubleshooting point: once the latent index is batched or enumerated, the correct operation is not just "put `state` inside brackets." The code also has to say which dimension is being selected and which dimensions should survive.

## A Small HMM Example

Here is the pattern in a toy latent-state model:

```python
import pyro
import pyro.distributions as dist
from pyro.ops.indexing import Vindex


def model(observations, transition, emission):
    state = pyro.sample("state_0", dist.Categorical(emission[0]))

    for t, obs in enumerate(observations[1:], start=1):
        logits = Vindex(transition)[t - 1, state, :]
        state = pyro.sample(f"state_{t}", dist.Categorical(logits=logits))

        obs_logits = Vindex(emission)[t, state, :]
        pyro.sample(f"obs_{t}", dist.Categorical(logits=obs_logits), obs=obs)
```

In a real model I would add shape comments next to `transition` and `emission`, because that is usually where mistakes start. Something as plain as `transition: [time, state, state]` is often enough to catch a wrong axis before inference fails.

## Possible Uses in Single-Cell RNA-seq

Single-cell RNA-seq models often have discrete biological labels inside a larger count model. A cell might belong to a latent cell state, a response class, a copy-number state, or a branch along a trajectory. Whenever that latent label chooses a row from a parameter tensor, the same indexing issue comes back.

| Modeling problem | Latent index | Tensor being selected | Why `Vindex` fits |
| --- | --- | --- | --- |
| Cell-state mixture model | Cell type, subtype, or expression program `z_cell` | `rate[batch, program, gene]` | `Vindex(rate)[batch_id, z_cell, :]` picks the gene-rate vector for each cell while keeping the cell and batch axes clear. |
| Pseudotime or time-course HMM | Differentiation state `state_t` | `transition[condition, time, state, state]` | The current state selects a transition row without mixing up condition, time, and enumeration axes. |
| Perturb-seq response model | Response class after perturbation | `effect[perturbation, response_class, gene]` | Observed perturbation labels choose one axis; the latent response class chooses another. |
| Expression-based CNV or chromosomal-state model | Hidden segment state | `emission[chromosome, cnv_state, gene_bin]` | A state-space model can select state-specific expression shifts along ordered genomic bins. |
| Hierarchical annotation | Coarse and fine labels | `probs[coarse_type, fine_type, gene_program]` | The fine label can depend on a sampled coarse label, and both labels may be enumerated. |

For scRNA-seq, I would still keep the likelihood biology front and center: library size, overdispersion, batch effects, and usually a negative-binomial count model. `Vindex` only helps with one specific part of that job: cleanly selecting parameters using discrete latent variables.

It also does not replace scvi-tools or other mature single-cell packages. I would reach for it when the model needs a custom discrete structure, such as switching dynamics, lineage uncertainty, perturbation response classes, or interpretable program mixtures.

## Practical Notes

- Use `Vindex` when a sampled discrete variable indexes a batched parameter tensor.
- Test under the same enumeration setting used for inference; scalar sampling can hide the bug.
- Keep event dimensions, such as genes in a count vector, out of the indexed batch shape.
- Write the shape next to every transition, emission, or component tensor.

## References

- [Pyro enumeration tutorial: indexing with enumerated variables](https://pyro.ai/examples/enumeration.html#indexing-with-enumerated-variables)
- [Pyro `Vindex` API documentation](https://docs.pyro.ai/en/stable/ops.html#pyro.ops.indexing.Vindex)
- [PyTorch tensor views note on basic versus advanced indexing](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [scvi-tools documentation](https://docs.scvi-tools.org/)
