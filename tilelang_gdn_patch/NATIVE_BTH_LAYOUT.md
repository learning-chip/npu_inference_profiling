# BTH vs BHT for TileLang on Ascend

**Triton / public API** use ``head_first=False``: tensors are ``[batch, seq, head, dim]`` (**BTH**).

The vendored ``opt_gdn`` TileLang kernels follow the original **head-first** convention
``[batch, head, seq, dim]`` (**BHT**). Along the sequence dimension ``L``, slices ``tensor[b, h, :]``
are **contiguous** in memory, which matches what Ascend ``T.copy`` paths expect for chunkwise loads.

Using **BTH** tensors directly inside those kernels would make the per-head sequence strided
(stride ``H * D`` between consecutive tokens), and in practice outputs diverged from the reference
when we only permuted index expressions without fixing DMA stride behavior.

So :mod:`tilelang_gdn_patch.pipeline` applies **one** ``transpose(1, 2).contiguous()`` on ``q, k, v, g, beta``
before kernels and **one** on the output ``o`` afterward. :mod:`tilelang_gdn_patch.api` stays free of
that transpose so all layout policy lives next to the TileLang chain.
