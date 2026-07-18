# CotabbyInference Coding Instructions

CotabbyInference is Cotabby's product-specific C++ wrapper around llama.cpp. Preserve its narrow
single-sequence boundary unless a shipping Cotabby requirement explicitly changes it.

## Ownership

- `CotabbyInferenceEngine` owns one model, one context, and at most one active sequence.
- Cotabby's Swift `LlamaRuntimeCore` owns generation serialization, token budgets, prompt/cache
  compatibility, streaming, and lifecycle admission.
- The native engine owns sampler state, seed/pending-token handoff, KV mutation, token masks,
  cancellation observation, and native resource release.
- Do not add editor, Accessibility, UI, or hosted-service concepts to this package.

## Native Safety

- Keep the public header free of llama.cpp and ggml types.
- Treat `SampleResult.piece` as borrowed memory that expires when sequence storage changes.
- Serialize every llama context mutation through `decode_mutex`.
- Keep cancellation atomic and nonblocking; an active `llama_decode` is not preemptible.
- Never destroy a sequence or unload the model while another non-cancellation operation is using it.
- Preserve the changing external sequence ID even though the internal llama sequence slot is fixed
  at zero; it protects replacement sequences from late cancellation.

## Change Strategy

- Add only APIs used by the current Cotabby integration or backed by a concrete consumer plan.
- Prefer removing obsolete experimental surfaces over retaining speculative generality.
- Treat the llama.cpp binary URL/checksum as a native compatibility boundary.
- Update `README.md`, package tests, and local `.internal/` notes when architecture changes.
- Keep `.internal/` gitignored; it is local study material, not PR content.

## Validation

Run `swift test` for compilation and no-model contracts. When a GGUF is available, also run:

~~~bash
COTABBY_TEST_MODEL_PATH=/absolute/path/model.gguf swift test
~~~

For public API changes, build the current Cotabby app against the local package before merging.
