"""TODO.
"""
import os
from pathlib import Path

from tokenizer import train_tokenizer
from transformers.tokenization_utils_base import BatchEncoding

test_batch = [
    "There are some unseen words here banana zen 😁 but also some seen text",
    "Lots of unknowns in this one I expect",
    "text and words and tokens and stuff"
]

test_tokenizer_path = "C:/Users/willf/DataScience/Repos/deep-embedding/de/vae/prepare/text/test_data/test_bpe_tokenizer.json"
base_data_path = Path("C:/Users/willf/DataScience/Repos/deep-embedding/de/vae/prepare/text/test_data")
data_paths = [str(path) for path in base_data_path.glob("**/*.txt")]

tk = train_tokenizer(data_paths=data_paths, save_path=test_tokenizer_path)


encoded_batch: BatchEncoding = tk.encode_batch(test_batch)

# def training_mask(sequence: str, mask_probability: float | int):


#     return sequence

# encoding.input_ids[0, 22:31] = tk.token_to_id("[MASK]")

detokenized = tk.decode_batch(sequences=[encoding.ids for encoding in encoded_batch], skip_special_tokens=False)

print(detokenized, "\n", test_batch)

os.remove(test_tokenizer_path)
