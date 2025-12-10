# RIFT
## License
Apache License 2.0
Patent pending

Realized Interference Fields for hallucination reduction in large language models.

Large language models often generate responses that are fluent but internally inconsistent or factually unstable, commonly referred to as hallucinations. Most existing approaches treat this as a knowledge or grounding problem.

RIFT treats hallucination as a stability problem: some candidate responses cannot be consistently maintained relative to prior context or their own implied commitments.

RIFT is a post-generation evaluation layer that:

- samples multiple candidate responses from a base model
- scores each candidate using a fixed energy function
- selects the response with the lowest instability

RIFT does not:

- modify model weights
- introduce task-specific heuristics or hard-coded facts
- perform learning or self-improvement
- act autonomously or plan
- persist memory across sessions

RIFT is model-agnostic and can be applied to any language model capable of producing multiple candidate responses. It operates entirely at response selection time and can be integrated without retraining or fine-tuning the underlying model.

In practice, RIFT prefers responses that remain consistent with prior conversation state. When all candidate responses are unstable, RIFT fails conservatively rather than fabricating additional structure.

## Documentation

- Whitepaper: [`whitepaper/rift_whitepaper.pdf`](./whitepaper/)
- Benchmarks: [`benchmarks/benchmarks.md`](./benchmarks/)

## Usage

RIFT is currently provided as a reference implementation and research prototype.

See `examples/` for minimal usage demonstrations.

## Status

This repository accompanies an initial public release of the RIFT mechanism. The implementation and benchmarks may evolve, but the core mechanism described in the whitepaper is considered stable.

## License

Apache 2.0. See `LICENSE` for details.


