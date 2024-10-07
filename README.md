# Overview
The task focuses on enhancing a basic GPT model by incorporating Rotary Position Embeddings (RoPE) and Grouped Query Attention (GQA), key components in state-of-the-art Large Language Models (LLMs) like LLAMA-2.

#Dataset
The dataset used for this homework is a collection of the complete works of Shakespeare, provided in input.txt. The dataset is approximately 1.1MB in size.

## Project Structure
```
RoPE-GQA/
│
├── chargpt.py             # Main script to train the transformer model.
├── input.txt              # Dataset (complete works of Shakespeare).
└── mingpt/
    ├── model.py           # Contains the model definition. Primary focus for RoPE and GQA.
    ├── trainer.py         # Code for the training loop of the model.
    └── utils.py           # Helper functions for saving logs and configs.
```
