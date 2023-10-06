# DeBERTinha - A DeBERTa V3 XSmall adapted to the Brazilian Portuguese language

We present an approach for adapting the DebertaV3 XSmall model pre-trained in English for Brazilian Portuguese natural language processing (NLP) tasks. A key aspect of the methodology involves a multistep training process to ensure the model is effectively tuned for the Portuguese language. Initial datasets from Carolina and BrWac are preprocessed to address issues like emojis, HTML tags, and encodings. A Portuguese-specific vocabulary of 50,000 tokens is created using SentencePiece. Rather than training from scratch, the weights of the pre-trained English model are used to initialize most of the network, with random embeddings, recognizing the expensive cost of training from scratch. The model is fine-tuned using the replaced token detection task in the same format of DebertaV3 training. The adapted model, called DeBERTinha, demonstrates effectiveness on downstream tasks like named entity recognition, sentiment analysis, and determining sentence relatedness, outperforming BERTimbau-Large in two tasks despite having only 40M parameters.

# A basic example on how we can use DeBERTinha

````python
from transformers import AutoTokenizer
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained('sagui-nlp/debertinha-ptbr-xsmall')
tokenizer = AutoTokenizer.from_pretrained('sagui-nlp/debertinha-ptbr-xsmall')

input_ids = tokenizer.encode('Tinha uma pedra no meio do caminho.', return_tensors='pt')

with torch.no_grad():
    outs = model(input_ids)
    encoded = outs.last_hidden_state[0, 1:-1]  # Ignore [CLS] and [SEP] special tokens
````

# Citation

Our paper can be found [here](https://arxiv.org/abs/2309.16844).
If you use this work please cite:

```
@misc{campiotti2023debertinha,
      title={DeBERTinha: A Multistep Approach to Adapt DebertaV3 XSmall for Brazilian Portuguese Natural Language Processing Task}, 
      author={Israel Campiotti and Matheus Rodrigues and Yuri Albuquerque and Rafael Azevedo and Alyson Andrade},
      year={2023},
      eprint={2309.16844},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
