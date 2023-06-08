import random
from collections import OrderedDict

import nltk
import numpy as np
import pandas as pd

import clip
from MONET.utils.static import concept_to_prompt


def str_to_token(caption_str, use_random=True):
    # tokenize
    while True:
        # print("use random", use_random)
        # print("try sampling", np.random.randint(10), random.randint(0, 9))
        try:
            caption_tokenized = clip.tokenize(caption_str, truncate=False)
        except RuntimeError:
            sentences = nltk.sent_tokenize(caption_str)  #
            words = nltk.word_tokenize(caption_str)
            if len(sentences) == 0:
                raise RuntimeError("No sentences found in caption")
            if len(words) == 0:
                raise RuntimeError("No words found in caption")

            if len(words) == 1:
                random_idx = np.random.randint(len(caption_str)) if use_random else 0
                # print(caption_save, caption_str)
                # print("caption_str random_idx", random_idx, len(caption_str))
                caption_str = caption_str[
                    random_idx : min(random_idx + len(caption_str) // 2, len(caption_str))
                ]
            elif len(sentences) == 1:
                random_idx = np.random.randint(len(words)) if use_random else 0
                # print("words random_idx", random_idx, len(words))
                words = words[random_idx : min(random_idx + len(words) // 2, len(words))]
                if words[-1] == ".":
                    caption_str = " ".join(words[:-1]) + "."
                else:
                    caption_str = " ".join(words)
            else:
                random_idx = np.random.randint(len(sentences)) if use_random else 0
                # print("sentence random_idx", random_idx, len(sentences))
                sentences = sentences[
                    random_idx : min(
                        random_idx + len(sentences) // 2,
                        len(sentences),
                    )
                ]
                caption_str = " ".join(sentences)
        else:
            break
    return caption_str, caption_tokenized


def generate_prompt_token_from_caption(self, idx, caption_col="caption_final", use_random=True):
    # to series
    caption_series = self.text_data.loc[self.metadata_all.iloc[idx].name]
    if isinstance(caption_series, pd.DataFrame):
        caption_series = caption_series.iloc[np.random.randint(len(caption_series))]
    assert isinstance(caption_series, pd.Series), f"Not a pandas Series: {caption_series}"
    caption_str = caption_series[caption_col]

    # to string
    if isinstance(caption_str, list):
        caption_str = caption_str[np.random.randint(len(caption_str))]

    assert isinstance(caption_str, str), f"Not a string: {caption_str}"

    caption_save = caption_str
    # caption_str.encode("ascii", errors="ignore").decode()
    # if caption_str == caption_save:
    #    print(caption_save, caption_str, "removed non-ascii characters")

    caption_str, caption_tokenized = str_to_token(caption_str, use_random=use_random)
    # if caption_save != caption_str:
    #     print(f"Caption was truncated")
    #     print(f"Original:  {caption_save})")
    #     print(f"Truncated: {caption_str})")
    return caption_str, caption_tokenized


def generate_prompt_token_from_concept(concept_name, use_random=True):
    prompt_dict, text_counter = concept_to_prompt(concept_name)

    caption_str_tokenized_dict = OrderedDict()
    for key, value in prompt_dict.items():
        # if key != "original":
        caption_str, caption_tokenized = str_to_token(value, use_random=use_random)
        caption_str_tokenized_dict[key] = (caption_str, caption_tokenized)
        # caption_str_tokenized_dict[caption_str] = caption_tokenized
        # .append((caption_str, caption_tokenized))
    return caption_str_tokenized_dict
