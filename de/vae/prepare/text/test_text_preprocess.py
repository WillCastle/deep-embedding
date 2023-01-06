"""TODO.
"""
from test_tokenizer import tk
from text_preprocess import TextPreprocessor

text_batch = [
    "There are some unseen words here banana zen 😁 but also some seen text",
    "text and words and tokens and stuff",
    "Lots of unknowns in this one I expect",
]
text_preprocessor = TextPreprocessor(tokenizer=tk)

print(text_preprocessor(text_batch))
