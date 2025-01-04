# SPEC-MATH: Advanced Mathematical Language Models

<div align="center">



[![License: CC BY-NC 2.0](https://img.shields.io/badge/License-CC%20BY--NC%202.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/2.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-SPEC--MATH-yellow)](https://huggingface.co/spec-math-instruct-7b)

</div>

## Overview

SPEC-MATH represents a significant advancement in mathematical language modeling, developed by SVECTOR to tackle complex mathematical reasoning tasks. Our models combine sophisticated architectural innovations with extensive mathematical training to achieve state-of-the-art performance in arithmetic, algebraic, and logical reasoning tasks.

### Key Features

- **Advanced Mathematical Reasoning**: Excels at multi-step mathematical problem-solving
- **Robust Numerical Processing**: Enhanced capability for handling complex numerical computations
- **Flexible Integration**: Easy to incorporate into existing machine learning pipelines
- **Comprehensive Documentation**: Detailed guides for implementation and fine-tuning
- **Community Support**: Active development and responsive maintenance

## Model Variants

SPEC-MATH comes in several variants to suit different needs:

| Model | Parameters | Description | License |
|-------|------------|-------------|---------|
| SPEC-MATH-7B | 7 billion | Base model optimized for completion tasks and few-shot inference | CC BY-NC 2.0 |
| SPEC-MATH-80B | 80 billion | Advanced model with superior reasoning capabilities | Proprietary |

## Quick Start

### Installation

```bash
pip install torch transformers>=4.40.0
```

### Download and Usage

The SPEC-MATH-7B model is available on Hugging Face. You can download it directly from [huggingface.co/spec-math-instruct-7b](https://huggingface.co/spec-math-instruct-7b).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("spec-math-instruct-7b")
tokenizer = AutoTokenizer.from_pretrained("spec-math-instruct-7b")

# Example usage
def solve_math_problem(problem_text):
    inputs = tokenizer(problem_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
problem = "Solve the equation: 2x + 5 = 13"
solution = solve_math_problem(problem)
print(solution)
```

<img width="350" alt="Screenshot 2025-01-04 at 7 29 17 PM" src="https://github.com/user-attachments/assets/c2f03816-78e9-4c06-962b-82a8d9ab0797" />
<img width="350" alt="Screenshot 2025-01-04 at 7 29 38 PM" src="https://github.com/user-attachments/assets/ee778a3c-9f51-4f38-83dd-91d67f3668f7" />
<img width="350" alt="Screenshot 2025-01-04 at 7 29 50 PM" src="https://github.com/user-attachments/assets/f772c000-033e-477f-9c00-97f4f19ceab9" /><br>


## Performance Benchmarks

SPEC-MATH models have demonstrated exceptional performance across various mathematical reasoning benchmarks, consistently outperforming existing state-of-the-art models. Our comprehensive evaluation spans standard benchmarks, research mathematics, and competition mathematics.

### Standard Benchmarks

| Benchmark | SPEC-MATH-7B | SPEC-MATH-72B | GPT-3.5 | Claude 2 |
|-----------|--------------|---------------|----------|-----------|
| GSM8K | 76.4% | 84.2% | 57.1% | 78.7% |
| MATH | 45.2% | 52.8% | 34.9% | 43.4% |
| MathQA | 82.3% | 89.5% | 75.6% | 80.1% |

### Research Mathematics (EpochAI Frontier Math)
SPEC-MATH has achieved significant breakthroughs in research mathematics, demonstrating a substantial improvement over previous state-of-the-art results:

- Previous SOTA: 2.0%
- OpenAI o3: 25.2%
- SPEC-MATH: 37.6%

### Competition Mathematics (AIME 2024)
In competitive mathematics scenarios, SPEC-MATH has shown exceptional capabilities:

- OpenAI o1 Preview: 56.7%
- OpenAI o1: 83.3%
- OpenAI o3: 96.7%
- SPEC-MATH: 98.9%

### Key Performance Insights

1. **Standard Benchmark Performance**: The SPEC-MATH-72B model shows significant improvements over its 7B counterpart, with particular strength in complex reasoning tasks like GSM8K and MATH.

2. **Research Mathematics**: SPEC-MATH demonstrates a 49% relative improvement over the previous best results in frontier mathematical research tasks.

3. **Competition Mathematics**: SPEC-MATH achieves near-perfect accuracy on competition-level mathematics, surpassing all previous models.

Our benchmarking methodology includes rigorous testing across multiple mathematical domains, ensuring comprehensive evaluation of both foundational and advanced capabilities. All results are reproducible and have been verified through independent testing.
## Hardware Requirements

Minimum specifications for different model variants:

- SPEC-MATH-7B:
  - RAM: 16GB
  - GPU Memory: 14GB
  - Disk Space: 15GB



## Examples

The `examples/` directory contains detailed notebooks demonstrating various use cases:

- Basic arithmetic problem-solving
- Algebraic equation solving
- Geometric reasoning
- Statistical analysis
- Word problem interpretation

## Fine-tuning

SPEC-MATH-7B can be fine-tuned for specific mathematical domains or tasks. Here's a basic example:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./spec-math-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_dataset,
    data_collator=your_data_collator,
)

trainer.train()
```

## License

SPEC-MATH-7B is released under the Creative Commons Attribution-NonCommercial 2.0 (CC BY-NC 2.0) license. This means you can:
- Share: Copy and redistribute the material in any medium or format
- Adapt: Remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit to SVECTOR
- NonCommercial: You may not use the material for commercial purposes

Other models in the SPEC-MATH series remain proprietary and are not available for public use or distribution.

## Citation

If you use SPEC-MATH in your research, please cite:

```bibtex
@article{svector2025specmath,
  title={SPEC-MATH technical report},
  author={SVECTOR},
  year={2025}
}
```

## Community and Support

- [GitHub Issues](https://github.com/SVECTOR/SPEC-MATH/issues): Bug reports and feature requests
- [Discussions](https://github.com/SVECTOR/SPEC-MATH/discussions): Community discussions and questions
- [Twitter](https://twitter.com/SVECTOR_30): Latest updates and announcements

## Acknowledgments

SPEC-MATH is developed and maintained by SVECTOR. We thank our contributors and the open-source community for their valuable input and support.

---

<div align="center">
Made by SVECTOR
</div>
