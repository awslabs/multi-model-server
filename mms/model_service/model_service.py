from abc import ABCMeta, abstractmethod, abstractproperty

URL_PREFIX = ('http://', 'https://', 's3://')


class ModelService(object):
    '''ModelService wraps up all preprocessing, inference and postprocessing
    functions used by model service. It is defined in a flexible manner to
    be easily extended to support different frameworks.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, path, ctx):
        self.context = ctx

    @abstractmethod
    def inference(self, data):
        pass

    @abstractmethod
    def ping(self):
        pass

    @abstractproperty
    def signature(self):
        pass


class SingleNodeService(ModelService):
    '''SingleNodeModel defines abstraction for model service which loads a
    single model.
    '''
    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : list of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        data = self._preprocess(data)
        data = self._inference(data)
        data = self._postprocess(data)
        return data

    @abstractmethod
    def _inference(self, data):
        '''
        Internal inference methods. Run forward computation and
        return output.

        Parameters
        ----------
        data : list of NDArray
            Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        '''
        return data

    def _preprocess(self, data):
        '''
        Internal preprocess methods. Do transformation on raw
        inputs and convert them to NDArray.

        Parameters
        ----------
        data : list of object
            Raw inputs from request.

        Returns
        -------
        list of NDArray
            Processed inputs in NDArray format.
        '''
        return data

    def _postprocess(self, data):
        '''
        Internal postprocess methods. Do transformation on inference output
        and convert them to MIME type objects.

        Parameters
        ----------
        data : list of NDArray
            Inference output.

        Returns
        -------
        list of object
            list of outputs to be sent back.
        '''
        return data


class MultiNodesService(ModelService):
    pass

