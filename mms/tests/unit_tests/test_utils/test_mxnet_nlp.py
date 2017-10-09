import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path + '/../../..')

import unittest
import utils.mxnet.nlp as nlp

class TestMXNetNLPUtils(unittest.TestCase):
    def test_encode_sentence(self):
        vocab = {}
        sentence = []
        for i in range(100):
            vocab['word%d' % (i)] = i
        sen_vec = [0, 56, 8, 10]
        for i in sen_vec:
            sentence.append('word%d' % (i))
        res1, out1 = nlp.encode_sentences([sentence], vocab)
        assert res1[0] == sen_vec, "encode_sentence method failed. " \
                                   "Result vector invalid."
        assert len(out1) == len(vocab), "encode_sentence method failed. " \
                                        "Generated vocab incorrect."

        res2, out2 = nlp.encode_sentences([sentence])
        assert res2[0] == [i for i in range(len(sentence))], \
            "encode_sentence method failed. Result vector invalid."
        assert len(out2) == len(sentence) + 1, "encode_sentence method failed. " \
                                               "Generated vocab incorrect."

