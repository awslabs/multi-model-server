import argparse


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try: 
          setattr(namespace, 'models', {kv.split('=', 1)[0]: kv.split('=', 1)[1] for kv in values})
        except Exception:
          raise Exception('Failed to parse <model=path>: ' + str(values) + 
                          ' Format should be <model-name>=<model-path> (Local file path, URL, S3).')
    
class ArgParser(object):
    '''Argument parser
    '''

class ArgParser(object):
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(prog='mxnet-model-serving', description='MXNet Model Serving')

        parser.add_argument('--models',
                            required=True,
                            action=StoreDictKeyPair,
                            metavar='KEY1=VAL1,KEY2=VAL2...',
                            nargs='+',
                            help='Models to be deployed')

        parser.add_argument('--process', help='Using user defined model service')

        parser.add_argument('--gen-api', help='Generate API')

        parser.add_argument('--port', help='Port')

        return parser.parse_args()

    @staticmethod
    def parse_export_args():
        parser_export = argparse.ArgumentParser(prog='model-export', description='MXNet Model Export')

        parser_export.add_argument('--model',
                                   required=True,
                                   metavar='KEY=VAL',
                                   help='Model to be exported. Key is model name. '
                                        'Value is path contains model files.')

        parser_export.add_argument('--signature',
                                   required=True,
                                   type=str,
                                   help='Path to signature file')

        parser_export.add_argument('--synset',
                                   type=str,
                                   help='Path to synset file')

        parser_export.add_argument('--export-path',
                                   type=str,
                                   help='Path to exported model')

        return parser_export.parse_args()


