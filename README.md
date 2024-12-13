# GPT2-from-Scratch

This repository contains a minimalist, educational implementation of a Generative Pre-trained Transformer (GPT) model—similar in spirit to GPT-2 architecture—constructed entirely from scratch. It aims to provide a transparent view of how the various components of a transformer-based language model fit together, including the tokenization, dataset creation, training loop, and inference steps.

## Features

- **Attention Mechanisms**:  
  Implements key building blocks of transformer architectures:
  - **Self-Attention** and **Causal Self-Attention**: Understanding how queries, keys, and values interact is crucial.  
  - **Multi-Head Attention (MHA)**: Parallel attention heads allow the model to focus on different positions and semantic relationships in the sequence simultaneously.

- **Transformer Components**:  
  Includes essential transformer components such as:
  - **Layer Normalization**  
  - **Feed-Forward Networks (FFN)**  
  - **Residual Connections**  
  - **Positional Embeddings**

- **GPT Architecture**:  
  A stack of transformer blocks is combined with token and positional embeddings to create a decoder-only GPT model. This model can be trained to predict the next token given previous tokens, enabling text generation capabilities.

- **Configurable Hyperparameters**:  
  A `CONFIG` dictionary at the end of the file lets you easily tweak model parameters (e.g., number of layers, embedding sizes, number of attention heads, context length, dropout rate) and training parameters (learning rate, batch size, number of epochs).

- **Dataset and Tokenization**:  
  Utilizes a dataset class `GPTDataset` that handles tokenization (with the GPT-2 tokenizer provided by `tiktoken`) and the creation of training examples. You can easily replace the training text and modify sequence lengths.

- **Training and Evaluation Scripts**:  
  Sample training loops and evaluation routines are included. Easily adapt the code to train on your own text data and monitor validation losses.

## Why Build Your Own GPT?

1. **Educational Purposes**: Gain a deeper understanding of the internals of transformer architectures and language models.
2. **Experimentation**: Try out modifications and customizations that may not be straightforward in large codebases or frameworks.
3. **Foundational Knowledge**: Master the components (attention, feed-forward networks, embeddings) and concepts (masking, positional encoding) that are fundamental to modern NLP.

## Getting Started

### Requirements

- Python 3.7+
- PyTorch (for GPU acceleration, if available)
- `tiktoken` for GPT-2 compatible tokenization

You can install necessary packages using:
```bash
pip install torch tiktoken
```

If you intend to train on GPU, ensure that you have CUDA-compatible PyTorch installed:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```
(Adjust CUDA version as needed.)

### Code Structure

All code has been consolidated into a single file here for demonstration purposes. In a more mature project, you might break it down into multiple files and directories:

- **Attention classes (`SelfAttention`, `CausalAttention`, `MultiHeadAttention`)**:  
  Core building blocks for the transformer mechanism.
  
- **Transformer components (`LayerNorm`, `FeedForward`, `TransformerBlock`)**:  
  Composed into a GPT model.
  
- **GPT Model (`GPT2`)**:  
  A decoder-only transformer stack similar to GPT-2. Embedding layers, positional embeddings, transformer blocks, and a final projection layer are included.
  
- **Dataset (`GPTDataset`)**:  
  Prepares tokenized sequences and sets up input and target tensors for next-token prediction tasks.
  
- **Configuration Dictionary (`CONFIG`)**:  
  Central place to store hyperparameters and model settings.

### Example Usage

#### Training

1. Prepare your training text file (`.txt`) and place it in the current directory or specify its path.
   
2. Modify `CONFIG` as desired (e.g., `vocab_size`, `context_length`, `batch`, `epoch`, etc.).

3. Run the training code block:
   ```bash
   python pretrain.py
   ```
   
   The script will:
   - Load and tokenize the training text.
   - Create `GPTDataset` for training and validation sets.
   - Initialize the `GPT2` model.
   - Run the training loop, periodically evaluating and printing losses.

#### Generating Text

Once the model is trained, you can generate new text by providing a prompt and calling the generation function:

```python
start_context = "The world is"
with torch.no_grad():
    token_ids = generate_text(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer).to('cuda'),
        max_new_tokens=50,
        context_length=CONFIG["context_length"]
    )
print("Generated Text:", token_ids_to_text(token_ids, tokenizer))
```

This will produce a continuation of the given prompt.

### Model Configuration

In the `CONFIG` dictionary, you can modify:

- **Model size**: `emb_dim`, `n_heads`, `n_layers`
- **Context length**: `context_length`
- **Vocabulary size**: `vocab_size` (depends on tokenizer)
- **Training params**: `batch`, `epoch`, `learning_rate`, `dropout_rate`

### Performance Considerations

- **GPU Recommended**: Training even a modest GPT model is computationally expensive. A GPU will drastically speed up the process.
- **Batch Size and Sequence Length**: Adjust them based on your hardware constraints.
- **Precision and Techniques**: Consider techniques like mixed-precision training (using `torch.cuda.amp`) for faster training and lower memory usage.

## Roadmap

- **More Advanced Tokenization**: Support for Byte Pair Encoding (BPE) or SentencePiece.
- **Scaling Up**: Adding gradient accumulation, mixed-precision training, and distributed training for larger models.
- **Metrics and Logging**: Integration of tensorboard or other logging tools for better visualization and debugging.
- **Sampling Strategies**: Implement top-k, top-p (nucleus) sampling, or temperature-based sampling for more interesting text generation.

## License

This project is provided under the MIT License. Feel free to use, modify, and distribute this code for educational or research purposes.

---

**Happy hacking!** Experiment, break things, and learn how modern language models come together under the hood.
