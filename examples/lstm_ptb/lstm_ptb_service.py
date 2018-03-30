import mxnet as mx
import os
import json

from mms.model_service.mxnet_model_service import MXNetBaseService
from mms.utils.mxnet import nlp


class MXNetLSTMService(MXNetBaseService):
    """LSTM service class. This service consumes a sentence
    from length 0 to 60 and generates a sentence with the same size.
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        self.model_name = model_name
        self.ctx = mx.gpu(int(gpu)) if gpu is not None else mx.cpu()
        signature_file_path = os.path.join(model_dir, manifest['Model']['Signature'])
        if not os.path.isfile(signature_file_path):
            raise RuntimeError('Signature file is not found. Please put signature.json '
                               'into the model file directory...' + signature_file_path)
        try:
            signature_file = open(signature_file_path)
            self._signature = json.load(signature_file)
        except:
            raise Exception('Failed to open model signature file: %s' % signature_file_path)
            
        self.data_names = []
        self.data_shapes = []
        for input in self._signature['inputs']:
            self.data_names.append(input['data_name'])
            self.data_shapes.append((input['data_name'], tuple(input['data_shape'])))
        
        # Load pre-trained lstm bucketing module
        load_epoch = 100
        num_layers = 2
        num_hidden = 200
        num_embed = 200

        self.buckets = [10, 20, 30, 40, 50, 60]
        self.start_label = 1
        self.invalid_key = '\n'
        self.invalid_label = 0
        self.layout = 'NT'

        vocab_dict_file = os.path.join(model_dir, "vocab_dict.txt")
        self.vocab = {}
        self.idx2word = {}
        with open(vocab_dict_file, 'r') as vocab_file:
            self.vocab[self.invalid_key] = self.invalid_label
            for line in vocab_file:
                word_index = line.split(' ')
                if len(word_index) < 2 or word_index[0] == '':
                    continue
                self.vocab[word_index[0]] = int(word_index[1].rstrip())
        for key, val in self.vocab.items():
            self.idx2word[val] = key

        stack = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, mode='lstm').unfuse()

        # Define symbol generation function for bucket module
        def sym_gen(seq_len):
            data = mx.sym.Variable('data')
            embed = mx.sym.Embedding(data=data, input_dim=len(self.vocab),
                                     output_dim=num_embed, name='embed')

            stack.reset()
            outputs, _ = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

            pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
            pred = mx.sym.FullyConnected(data=pred, num_hidden=len(self.vocab), name='pred')
            pred = mx.sym.softmax(pred, name='softmax')

            return pred, ('data',), None

        # Create bucketing module and load weights
        self.mx_model = mx.mod.BucketingModule(
            sym_gen=sym_gen,
            default_bucket_key=max(self.buckets),
            context=self.ctx)
        
        self.mx_model.bind(data_shapes=self.data_shapes, for_training=False)
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(stack, '%s/%s' % (model_dir, model_name), load_epoch)
        self.mx_model.set_params(arg_params, aux_params)

    def _preprocess(self, data):
        # Convert a string of sentence to a list of string
        sent = data[0][0]['input_sentence'].lower().split(' ')
        assert len(sent) <= self.buckets[-1], "Sentence length must be no greater than %d." % (self.buckets[-1])
        # Encode sentence to a list of int
        res, _ = nlp.encode_sentences([sent], vocab=self.vocab, start_label=self.start_label, invalid_label=self.invalid_label)

        return res

    def _inference(self, data):
        data_batch = nlp.pad_sentence(data[0], self.buckets, invalid_label=self.invalid_label,
                                      data_name=self.data_names[0], layout=self.layout)
        self.mx_model.forward(data_batch)
        return self.mx_model.get_outputs()

    def _postprocess(self, data):
        # Generate predicted sentences
        word_idx = mx.nd.argmax(data[0], axis=1).asnumpy()
        res = ''
        for idx in word_idx:
            res += self.idx2word[idx] + ' '
        return res
