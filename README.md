# Orion Atlas

**Cendrix AI** | [HuggingFace](https://huggingface.co/asvuep) | [cendrix.ai](https://cendrix.ai)

Novel LLM architecture family combining **Mamba-2 Hybrid SSM** + **Differential Attention** — the first known combination of these techniques at 7B scale.

## Models

| Model | Params | Context | Status |
|---|---|---|---|
| Orion Atlas 1B | 1.15B | 4K | Training |
| Orion Atlas 7B | 8.77B | 128K | Architecture released |

## Architecture Highlights (7B)
- 32 layers: 25 Mamba-2 SSM + 7 Differential Attention
- 128K token context window
- Purpose-built for agentic AI and tool calling
- First known Mamba-2 Hybrid + Differential Attention combination

## Quick Start
```python
# Coming with weights release
from model import OrionModel, OrionConfig, CONFIGS
config = CONFIGS["1B"]
model = OrionModel(config)
```

## Paper
arXiv: [coming soon]

## Citation
```bibtex
@misc{palermini2026orionatlas,
  title={Orion Atlas: A Mamba-2 Hybrid Architecture with Differential Attention for Agentic Language Models},
  author={Avery Palermini},
  year={2026},
  institution={Cendrix AI},
}
```

## License
Apache 2.0
