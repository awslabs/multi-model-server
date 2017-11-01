# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from mxnet import rnn

def encode_sentences(sentences, vocab=None, invalid_label=-1, invalid_key='\n', start_label=0):
    """Encode sentences and (optionally) build a mapping
    from string tokens to integer indices. Unknown keys
    will be added to vocabulary.

    Parameters
    ----------
    sentences : list of list of str
        A list of sentences to encode. Each sentence
        should be a list of string tokens.
    vocab : None or dict of str -> int
        Optional input Vocabulary
    invalid_label : int, default -1
        Index for invalid token, like <end-of-sentence>
    invalid_key : str, default '\\n'
        Key for invalid token. Use '\\n' for end
        of sentence by default.
    start_label : int
        lowest index.

    Returns
    -------
    result : list of list of int
        encoded sentences
    vocab : dict of str -> int
        result vocabulary
    """
    idx = start_label
    if vocab is None:
        vocab = {invalid_key: invalid_label}
        new_vocab = True
    else:
        new_vocab = False
    res = []
    for sent in sentences:
        coded = []
        for word in sent:
            if word not in vocab:
                if not new_vocab:
                    coded.append(invalid_label)
                    continue
                else:
                    if idx == invalid_label:
                        idx += 1
                    vocab[word] = idx
                    idx += 1
            coded.append(vocab[word])
        res.append(coded)

    return res, vocab