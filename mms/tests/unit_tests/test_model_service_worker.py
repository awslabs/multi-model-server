# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
ModelServiceWorker is the worker that is started by the MMS front-end.
"""

# pylint: disable=redefined-builtin

import socket
from collections import namedtuple

import mock
import pytest
from mms.model_service_worker import MXNetModelServiceWorker, MAX_FAILURE_THRESHOLD
from mms.mxnet_model_service_error import MMSError
from mms.service import Service
from mms.utils.model_server_error_codes import ModelServerErrorCodes as Err
from mock import Mock


@pytest.fixture()
def socket_patches(mocker):
    Patches = namedtuple('Patches', ['socket', 'log_msg', 'msg_validator', 'log_error', 'codec', 'json_load'])
    mock_patch = Patches(mocker.patch('socket.socket'),
                         mocker.patch('mms.model_service_worker.log_msg'),
                         mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators'),
                         mocker.patch('mms.model_service_worker.log_error'),
                         mocker.patch('mms.model_service_worker.OtfCodecHandler'),
                         mocker.patch('json.loads'))
    mock_patch.socket.recv.return_value = b'{}\r\n'
    return mock_patch


@pytest.fixture()
def model_service_worker(socket_patches):
    model_service_worker = MXNetModelServiceWorker('unix', 'my-socket', None, None)
    model_service_worker.sock = socket_patches.socket
    model_service_worker.service = Service('name', 'mpath', 'testmanifest', None, 0, 1)
    return model_service_worker


# noinspection PyClassHasNoInit
class TestCreateAndSendResponse:
    message = 'hello socket'
    code = 7
    resp = {'code': code, 'message': message}

    @pytest.fixture()
    def get_send_response_spy(self, model_service_worker, mocker):
        return mocker.patch.object(model_service_worker, 'send_response', wraps=model_service_worker.send_response)

    def test_with_preds(self, socket_patches, model_service_worker, get_send_response_spy):
        model_service_worker.create_and_send_response(socket_patches.socket, self.code, self.message)
        socket_patches.codec.create_response.return_value = 'Hello World'
        get_send_response_spy.assert_called()

        preds = "some preds"
        self.resp['predictions'] = preds
        model_service_worker.create_and_send_response(socket_patches.socket, self.code, self.message, preds)
        get_send_response_spy.assert_called()

        del (self.resp['predictions'])

    def test_with_exception(self, socket_patches, model_service_worker, get_send_response_spy):
        message = 'hello socket'
        code = 7
        socket_patches.codec.create_response.return_value = b'{}\r\n'
        get_send_response_spy.side_effect = Exception('Some Exception')
        with pytest.raises(Exception):
            model_service_worker.create_and_send_response(socket_patches.socket, code, message)

        socket_patches.log_error.assert_called()


# noinspection PyClassHasNoInit
class TestSendResponse:

    def test_with_io_error(self, socket_patches, model_service_worker):
        io_error = IOError("IO Error")
        socket_patches.socket.send.side_effect = io_error
        msg = 'hello socket'

        log_call_param = "{}: Send failed. {}.\nMsg: {}".format(Err.SEND_MSG_FAIL, repr(io_error),
                                                                msg)

        model_service_worker.send_failures = 0

        with pytest.raises(SystemExit) as ex:
            for i in range(1, MAX_FAILURE_THRESHOLD + 1):
                model_service_worker.send_response(socket_patches.socket, msg)
                socket_patches.socket.send.assert_called()
                assert model_service_worker.send_failures == i
                socket_patches.log_error.assert_called_with(log_call_param)

        # The exit status is exit(SEND_FAILS_EXCEEDS_LIMITS)
        assert ex.value.args[0] == Err.SEND_FAILS_EXCEEDS_LIMITS

    def test_with_os_error(self, socket_patches, model_service_worker):
        os_error = OSError("OS Error")
        socket_patches.socket.send.side_effect = os_error
        msg = 'hello socket'

        log_call_param = "{}: Send failed. {}.\nMsg: {}".format(Err.SEND_MSG_FAIL, repr(os_error), msg)

        model_service_worker.send_failures = 0

        with pytest.raises(SystemExit) as ex:
            for i in range(1, MAX_FAILURE_THRESHOLD + 1):
                model_service_worker.send_response(socket_patches.socket, msg)
                socket_patches.socket.send.assert_called()
                assert model_service_worker.send_failures == i
                socket_patches.log_error.assert_called_with(log_call_param)

        # The exit status is exit(SEND_FAILS_EXCEEDS_LIMITS)
        assert ex.value.args[0] == Err.SEND_FAILS_EXCEEDS_LIMITS


# noinspection PyClassHasNoInit
class TestRunServer:
    accept_result = ('cl_sock', None)

    def test_with_socket_bind_error(self, socket_patches, model_service_worker):
        bind_exception = socket.error("binding error")
        socket_patches.socket.bind.side_effect = bind_exception
        with pytest.raises(MMSError) as excinfo:
            model_service_worker.run_server()

        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_not_called()
        assert excinfo.value.get_code() == Err.SOCKET_BIND_ERROR

    def test_with_timeout(self, socket_patches, model_service_worker):
        exception = socket.timeout("Some Exception")
        socket_patches.socket.accept.side_effect = exception

        with pytest.raises(SystemExit):
            model_service_worker.run_server()
        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_called()
        socket_patches.socket.accept.assert_called()
        socket_patches.log_error.assert_called_with("Backend worker did not receive connection from frontend")

    def test_with_exception(self, socket_patches, model_service_worker):
        exception = Exception("Some Exception")
        socket_patches.socket.accept.side_effect = exception

        with pytest.raises(Exception):
            model_service_worker.run_server()
        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_called()
        socket_patches.socket.accept.assert_called()
        socket_patches.log_msg.assert_called_with("Waiting for a connection")

    def test_with_exception_debug(self, socket_patches, model_service_worker, mocker):
        exception = Exception("Some Exception")
        socket_patches.socket.accept.side_effect = exception
        mocker.patch('mms.model_service_worker.debug', True)
        model_service_worker.handle_connection = Mock()
        socket_patches.log_error.side_effect = exception

        with pytest.raises(Exception):
            model_service_worker.run_server()

        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_called()
        socket_patches.socket.accept.assert_called()
        socket_patches.log_error.assert_called_with("Backend worker error Some Exception")
        socket_patches.log_msg.assert_called_with("Waiting for a connection")

    def test_success_debug(self, socket_patches, model_service_worker, mocker):
        model_service_worker.sock.accept.side_effect = [self.accept_result, Exception()]
        model_service_worker.handle_connection = Mock()
        mocker.patch('mms.model_service_worker.debug', True)
        socket_patches.log_error.side_effect = Exception()

        with pytest.raises(Exception):
            model_service_worker.run_server()
        assert model_service_worker.sock.accept.call_count == 2
        model_service_worker.handle_connection.assert_called_once()

    # noinspection PyUnusedLocal
    def test_success(self, socket_patches, model_service_worker):
        model_service_worker.sock.accept.return_value = self.accept_result
        model_service_worker.handle_connection = Mock()
        with pytest.raises(SystemExit):
            model_service_worker.run_server()
        model_service_worker.sock.accept.assert_called_once()
        model_service_worker.handle_connection.assert_called_once()


# noinspection PyClassHasNoInit
class TestMXNetModelServiceWorker:
    # noinspection PyClassHasNoInit
    class TestInit:

        socket_name = "sampleSocketName"

        def test_missing_socket_name(self):
            with pytest.raises(ValueError) as excinfo:
                MXNetModelServiceWorker()
            assert excinfo.value.args[0] == 'Incomplete data provided'

        def test_socket_in_use(self, mocker):
            unlink = mocker.patch('os.unlink')
            pathexists = mocker.patch('os.path.exists')
            unlink.side_effect = OSError()
            pathexists.return_value = True

            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker('unix', self.socket_name)
            assert self.socket_name in excinfo.value.message
            assert excinfo.value.code == Err.SOCKET_ERROR
            assert excinfo.value.message == 'socket already in use: sampleSocketName.'

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['unlink', 'log', 'socket'])
            patches = Patches(
                mocker.patch('os.unlink'),
                mocker.patch('mms.model_service_worker.log_msg'),
                mocker.patch('socket.socket')
            )
            return patches

        @pytest.mark.parametrize('exception', [IOError('testioerror'), OSError('testoserror')])
        def test_socket_init_exception(self, patches, exception):
            patches.socket.side_effect = exception
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker('unix', self.socket_name)
            assert excinfo.value.code == Err.SOCKET_ERROR
            assert excinfo.value.message == 'Socket error in init sampleSocketName. {}'.format(repr(exception))

        def test_socket_unknown_exception(self, patches):
            patches.socket.side_effect = Exception('unknownException')
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker('unix', self.socket_name)
            assert excinfo.value.code == Err.UNKNOWN_EXCEPTION
            assert excinfo.value.message == "Exception('unknownException',)"

        def test_success(self, patches):
            MXNetModelServiceWorker('unix', self.socket_name)
            patches.unlink.assert_called_once_with(self.socket_name)
            patches.log.assert_called_once_with('Listening on port: sampleSocketName\n')
            patches.socket.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)

    # noinspection PyClassHasNoInit
    class TestLoadModel:

        data = {'modelPath': b'mpath', 'modelName': b'name', 'handler': b'handled'}

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['validate', 'loader'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators.validate_load_message'),
                mocker.patch('mms.model_loader.LegacyModelLoader.load')
            )
            return patches

        @pytest.fixture()
        def worker(self, mocker):
            mocker.patch.object(MXNetModelServiceWorker, 'service_manager', create=True),
            return object.__new__(MXNetModelServiceWorker)

        def test_load_value_error(self, patches, worker):
            patches.loader.side_effect = ValueError('verror')
            with pytest.raises(MMSError) as excinfo:
                worker.load_model(self.data)
            assert excinfo.value.code == Err.VALUE_ERROR_WHILE_LOADING
            assert excinfo.value.message == 'verror'

        def test_pass_mms_error(self, patches, worker):
            error = MMSError(Err.UNKNOWN_COMMAND, 'testerr')
            patches.loader.side_effect = error
            with pytest.raises(MMSError) as excinfo:
                worker.load_model(self.data)
            assert excinfo.value == error

        def test_unknown_error(self, patches, worker):
            patches.loader.side_effect = Exception('testerr')
            with pytest.raises(MMSError) as excinfo:
                worker.load_model(self.data)
            assert excinfo.value.code == Err.UNKNOWN_EXCEPTION_WHILE_LOADING
            assert excinfo.value.args[0] == "Exception('testerr',)"

        # noinspection PyUnusedLocal
        @pytest.mark.parametrize('batch_size', [(None, None), ('1', 1)])
        @pytest.mark.parametrize('gpu', [(None, None), ('2', 2)])
        def test_optional_args(self, patches, worker, batch_size, gpu):
            data = self.data.copy()
            if batch_size[0]:
                data['batchSize'] = batch_size[0]
            if gpu[0]:
                data['gpu'] = gpu[0]
            worker.load_model(data)

        def test_success(self, patches, worker):
            service, msg, code = worker.load_model(self.data)
            patches.validate.assert_called_once_with(self.data)
            patches.loader.assert_called_once_with('name', 'mpath', 'handled', None, None)
            assert msg == 'loaded model name'
            assert code == 200


# noinspection PyClassHasNoInit
class TestHandleConnection:
    mock_error = MMSError(Err.SEND_FAILS_EXCEEDS_LIMITS, "Unknown Error")

    @pytest.fixture()
    def get_spies(self, model_service_worker, mocker):
        Patches = namedtuple('Patches', ['codec', 'load_model', 'create_and_send_response'])
        mock_patch = Patches(mocker.patch('mms.model_service_worker.OtfCodecHandler'),
                             mocker.patch.object(model_service_worker, 'load_model',
                                                 wraps=model_service_worker.load_model),
                             mocker.patch.object(model_service_worker, 'create_and_send_response',
                                                 wraps=model_service_worker.create_and_send_response))
        model_service_worker.codec = mock_patch.codec
        return mock_patch

    def test_with_load(self, model_service_worker, socket_patches, get_spies):
        get_spies.codec.retrieve_msg.side_effect = [('load', 'somedata'), self.mock_error]

        model_service_worker.handle_connection(socket_patches.socket)

        get_spies.load_model.assert_called()
        get_spies.codec.retrieve_msg.assert_called()

    def test_with_unknown_cmd(self, model_service_worker, socket_patches, get_spies):
        result = "Received unknown command: {}".format('unk')
        err_code = Err.UNKNOWN_COMMAND

        get_spies.codec.retrieve_msg.side_effect = [('unk', 'somedata'), self.mock_error]

        model_service_worker.handle_connection(socket_patches.socket)

        get_spies.create_and_send_response.assert_called_with(socket_patches.socket, err_code, result, None)
        get_spies.codec.retrieve_msg.assert_called()

    def test_with_mms_error(self, model_service_worker, socket_patches, get_spies):
        get_spies.codec.retrieve_msg.side_effect = self.mock_error

        model_service_worker.handle_connection(socket_patches.socket)
        socket_patches.log_error.assert_called()

    def test_with_mms_unknown_error(self, model_service_worker, socket_patches, get_spies):
        error = Exception("Unknown Error")
        get_spies.codec.retrieve_msg.side_effect = [error, self.mock_error]

        model_service_worker.handle_connection(socket_patches.socket)

        socket_patches.log_error.assert_called()
        get_spies.create_and_send_response.assert_called()
