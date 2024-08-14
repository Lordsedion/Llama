# LLAMA

This is an implementation of the llama paper released by Meta Feb 2023 https://arxiv.org/abs/2302.13971.

In this repo, I implemented the entire LLama architecture from scratch and trained the model on different sizes of tokens to see the performance of the loss relative to 
the number of tokens used for training.

Token sizes used were: 50m, 100m, 200m, and 385m respectively.

The model configuration was the same for each token size.

## Llama 14m - 50m tokens

![download](https://github.com/user-attachments/assets/d3780756-c4cd-49c4-9293-a7c2eb473909)

14 million parameter LLama model trained on 50 million tokens.

## LLama 15m - 200m tokens

![download](https://github.com/user-attachments/assets/98f54a85-8f98-4611-9c55-eb96a1985271)

15 million parameter model trained on 200 million tokens achieving a loss of 5.422 using the tiktoken GPT-4 tokenizer
