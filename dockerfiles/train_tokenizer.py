from transformers import LongformerTokenizerFast
from datasets import load_dataset
import ftfy
from tqdm.auto import tqdm

tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")

dataset = load_dataset("wikipedia", language="pt", date="20230401",  beam_runner='DirectRunner')
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

def batch_iterator(batch_size=100):
    for i in tqdm(range(0, len(dataset), batch_size)):
        text = dataset[i : i + batch_size]["text"]
        text = [ftfy.fix_text(t) for t in text]
        yield text

new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=tokenizer.vocab_size)
new_tokenizer.save_pretrained("/longformer-pt-tokenizer")
