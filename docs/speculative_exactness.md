# Speculative decoding exactness gates

## Active guard: Qwen3.6 dense 27B affine Q8/group-64 (#2508)

oMLX does not admit the known affected Qwen3.6 dense-27B configuration to
native MTP or external VLM-MTP multi-token target verification.  It uses
ordinary decode instead.  The guard matches the public text-model geometry and
Q8/group-64 affine metadata; it does not depend on a directory name or on the
presence of native MTP weights.

This is deliberately narrow.  It does **not** block BF16, Q4/Q6, Qwen3.6 MoE,
other Qwen configurations, or DFlash.  Those paths need their own evidence;
the #2508 measurements only establish greedy divergence for the affected
target running native MTP or an external VLM-MTP drafter.

The policy is a safety gate, not a numerical fix.  It was added because the
reported temperature-0 comparison reproduced divergent tokens in both native
and external MTP verify loops, while BF16 did not.  CPU tests prove only that
the production admission predicates classify configuration metadata as
intended.

Removing or narrowing this guard requires a fresh GPU receipt on the exact
target path.  The receipt must compare ordinary decode with the candidate
speculative mode from equivalent prompt/cache state, with temperature 0,
matched seed, no cached tokens, and enough generated tokens to cover the
previous late stock-Q8 failure (at least 111 tokens).  It must report the
first differing token or byte-identical output; a successful load, high
acceptance rate, or a short generation is not sufficient evidence.
