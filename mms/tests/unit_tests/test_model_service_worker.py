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
Communication message format: JSON message
"""

# pylint: disable=redefined-builtin

import json
import socket
from collections import namedtuple

import pytest
from mms.model_service_worker import MXNetModelServiceWorker, MAX_FAILURE_THRESHOLD, emit_metrics
from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as err
from mock import Mock


@pytest.fixture()
def socket_patches(mocker):
    Patches = namedtuple('Patches', ['socket', 'log_msg', 'msg_validator', 'codec_helper', 'json_load', 'log_error'])
    mock_patch = Patches(mocker.patch('socket.socket'), mocker.patch('mms.model_service_worker.log_msg'),
                         mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators'),
                         mocker.patch('mms.model_service_worker.ModelWorkerCodecHelper'),
                         mocker.patch('json.loads'), mocker.patch('mms.model_service_worker.log_error'))
    mock_patch.socket.recv.return_value = b'{}\r\n'
    return mock_patch


@pytest.fixture()
def model_service_worker(socket_patches):
    model_service_worker = MXNetModelServiceWorker('unix', 'my-socket', None, None)
    model_service_worker.sock = socket_patches.socket

    return model_service_worker


def test_retrieve_model_input(socket_patches, model_service_worker):
    valid_inputs = [{'encoding': 'base64', 'value': 'val1', 'name': 'model_input_name'}]

    socket_patches.codec_helper.decode_msg.return_value = "some_decoded_resp"

    expected_response = {'model_input_name': 'some_decoded_resp'}

    model_in = model_service_worker.retrieve_model_input(valid_inputs)

    socket_patches.msg_validator.validate_predict_inputs.assert_called()
    socket_patches.codec_helper.decode_msg.assert_called()

    assert expected_response == model_in


class TestCreateAndSendResponse:

    message = 'hello socket'
    code = 7
    resp = {'code': code, 'message': message}

    @pytest.fixture()
    def get_send_response_spy(self, model_service_worker, mocker):
        return mocker.patch.object(model_service_worker, 'send_response', wraps=model_service_worker.send_response)

    def test_with_preds(self, socket_patches, model_service_worker, get_send_response_spy):

        model_service_worker.create_and_send_response(socket_patches.socket, self.code, self.message)
        get_send_response_spy.assert_called_with(socket_patches.socket, json.dumps(self.resp))

        preds = "some preds"
        self.resp['predictions'] = preds
        model_service_worker.create_and_send_response(socket_patches.socket, self.code, self.message, preds)
        get_send_response_spy.assert_called_with(socket_patches.socket, json.dumps(self.resp))

        del(self.resp['predictions'])

    def test_with_exception(self, socket_patches, model_service_worker, get_send_response_spy):
        message = 'hello socket'
        code = 7

        get_send_response_spy.side_effect = Exception('Some Exception')
        with pytest.raises(Exception):
            model_service_worker.create_and_send_response(socket_patches.socket, code, message)

        socket_patches.log_error.assert_called()


class TestRecvMsg:

    def test_with_nil_pkt(self, socket_patches):
        socket_patches.socket.recv.return_value = None
        with pytest.raises(SystemExit) as ex:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert ex.value.args[0] == 1  # The exit status is exit(1)

    @pytest.mark.parametrize('error', [OSError('oserr'), IOError('ioerr')])
    def test_with_sock_err(self, socket_patches, error):
        socket_patches.socket.recv.side_effect = error

        with pytest.raises(MMSError) as ex:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert ex.value.get_code() == err.RECEIVE_ERROR
        assert ex.value.get_message() == "{}".format(repr(error))

    def test_with_exception(self, socket_patches):
        socket_patches.socket.recv.side_effect = Exception('Some Exception')

        with pytest.raises(MMSError) as ex:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert ex.value.get_code() == err.UNKNOWN_EXCEPTION
        assert ex.value.get_message() == "Some Exception"

    def test_with_json_value_error(self, socket_patches):
        err_msg = "Some random json error"
        socket_patches.json_load.side_effect = ValueError(err_msg)

        with pytest.raises(MMSError) as ex:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert ex.value.get_code() == err.INVALID_REQUEST
        assert ex.value.get_message() == "JSON message format error: {}".format(err_msg)

    def test_with_missing_command(self, socket_patches):
        socket_patches.json_load.return_value = {}

        with pytest.raises(MMSError) as excinfo:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert excinfo.value.get_code() == err.INVALID_COMMAND
        assert excinfo.value.get_message() == "Invalid message received"

    def test_return_value(self, socket_patches):
        recv_pkt = {'command': {'Some command'}}
        socket_patches.json_load.return_value = recv_pkt

        cmd, data = MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        socket_patches.json_load.assert_called()
        assert 'command' in data.keys()
        assert cmd == recv_pkt['command']
        assert data == recv_pkt

    def test_loop(self, socket_patches):
        socket_patches.socket.recv.side_effect = [b"{}", b"{}\r\n"]
        recv_pkt = {'command': {'Some command'}}
        socket_patches.json_load.return_value = recv_pkt

        cmd, data = MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        socket_patches.json_load.assert_called()
        assert 'command' in data.keys()
        assert cmd == recv_pkt['command']
        assert data == recv_pkt


class TestSendResponse:

    def test_with_io_error(self, socket_patches, model_service_worker):
        io_error = IOError("IO Error")
        socket_patches.socket.send.side_effect = io_error
        msg = 'hello socket'

        log_call_param = "{}: Send failed. {}.\nMsg: {}".format(err.SEND_MSG_FAIL, repr(io_error),
                                                                ''.join([msg, '\r\n']))

        model_service_worker.send_failures = 0

        with pytest.raises(SystemExit) as ex:
            for i in range(1, MAX_FAILURE_THRESHOLD + 1):

                model_service_worker.send_response(socket_patches.socket, msg)
                socket_patches.socket.send.assert_called()
                assert model_service_worker.send_failures == i
                socket_patches.log_error.assert_called_with(log_call_param)

        # The exit status is exit(SEND_FAILS_EXCEEDS_LIMITS)
        assert ex.value.args[0] == err.SEND_FAILS_EXCEEDS_LIMITS

    def test_with_os_error(self, socket_patches, model_service_worker):
        os_error = OSError("OS Error")
        socket_patches.socket.send.side_effect = os_error
        msg = 'hello socket'

        log_call_param = "{}: Send failed. {}.\nMsg: {}".format(err.SEND_MSG_FAIL, repr(os_error),
                                                                ''.join([msg, '\r\n']))

        model_service_worker.send_failures = 0

        with pytest.raises(SystemExit) as ex:
            for i in range(1, MAX_FAILURE_THRESHOLD + 1):

                model_service_worker.send_response(socket_patches.socket, msg)
                socket_patches.socket.send.assert_called()
                assert model_service_worker.send_failures == i
                socket_patches.log_error.assert_called_with(log_call_param)

        # The exit status is exit(SEND_FAILS_EXCEEDS_LIMITS)
        assert ex.value.args[0] == err.SEND_FAILS_EXCEEDS_LIMITS


class TestRunServer:

    accept_result = ('cl_sock', None)

    def test_with_socket_bind_error(self, socket_patches, model_service_worker):
        bind_exception = socket.error("binding error")
        socket_patches.socket.bind.side_effect = bind_exception
        with pytest.raises(MMSError) as excinfo:
            model_service_worker.run_server()

        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_not_called()
        assert excinfo.value.get_code() == err.SOCKET_BIND_ERROR

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

    def test_success(self, socket_patches, model_service_worker):
        model_service_worker.sock.accept.return_value = self.accept_result
        model_service_worker.handle_connection = Mock()
        with pytest.raises(SystemExit):
            model_service_worker.run_server()
        model_service_worker.sock.accept.assert_called_once()
        model_service_worker.handle_connection.assert_called_once()


class TestMXNetModelServiceWorker:

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
            assert excinfo.value.code == err.SOCKET_ERROR
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
            assert excinfo.value.code == err.SOCKET_ERROR
            assert excinfo.value.message == 'Socket error in init sampleSocketName. {}'.format(repr(exception))

        def test_socket_unknown_exception(self, patches):
            patches.socket.side_effect = Exception('unknownException')
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker('unix', self.socket_name)
            assert excinfo.value.code == err.UNKNOWN_EXCEPTION
            assert excinfo.value.message == "Exception('unknownException',)"

        def test_success(self, patches):
            MXNetModelServiceWorker('unix', self.socket_name)
            patches.unlink.assert_called_once_with(self.socket_name)
            patches.log.assert_called_once_with('Listening on port: sampleSocketName\n')
            patches.socket.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)

    class TestCreatePredictResponse:

        sample_ret = ['val1']
        req_id_map = {0: 'reqId1'}
        empty_invalid_reqs = dict()
        invalid_reqs = {'reqId1': 'invalidCode1'}

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['encode'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.ModelWorkerCodecHelper.encode_msg')
            )
            return patches

        @pytest.fixture()
        def worker(self, mocker):
            return object.__new__(MXNetModelServiceWorker)

        def test_codec_exception(self, patches, worker):
            patches.encode.side_effect = Exception('testerr')
            with pytest.raises(MMSError) as excinfo:
                worker.create_predict_response(self.sample_ret, self.req_id_map, self.empty_invalid_reqs)
            assert excinfo.value.code == err.CODEC_FAIL
            assert excinfo.value.message == "codec failed Exception('testerr',)"

        @pytest.mark.parametrize('value', [(b'test', b'test'), ('test', b'test'), ({'test': True}, b'{"test": true}')])
        def test_value_types(self, patches, value, worker):
            ret = [value[0]]
            resp = worker.create_predict_response(ret, self.req_id_map, self.empty_invalid_reqs)
            patches.encode.assert_called_once_with('base64', value[1])
            assert set(resp[0].keys()) == {'requestId', 'code', 'value', 'encoding'}

        @pytest.mark.parametrize('invalid_reqs,requestId,code,value,encoding', [
            (dict(), 'reqId1', 200, 'encoded', 'base64'),
            ({'reqId1': 'invalidCode1'}, 'reqId1', 'invalidCode1', 'encoded', 'base64')
        ])
        def test_with_or_without_invalid_reqs(self, patches, invalid_reqs, requestId, code, value, encoding, worker):
            patches.encode.return_value = 'encoded'
            resp = worker.create_predict_response(self.sample_ret, self.req_id_map, invalid_reqs)
            assert resp == [{'requestId': requestId, 'code': code, 'value': value, 'encoding': encoding}]

    class TestRetrieveDataForInference:

        valid_req = [{'requestId': '111-222-3333', 'encoding': 'None|base64|utf-8', 'modelInputs': '[{}]'}]

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['validate'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators.validate_predict_data')
            )
            return patches

        @pytest.fixture()
        def worker(self, mocker):
            mocker.patch.object(MXNetModelServiceWorker, 'retrieve_model_input')
            return object.__new__(MXNetModelServiceWorker)

        def test_with_nil_request(self, patches, worker):
            with pytest.raises(ValueError) as excinfo:
                worker.retrieve_data_for_inference(requests=None)

            assert excinfo.value.args[0] == "Received invalid inputs"

        def test_with_invalid_req(self, patches, worker):
            worker.retrieve_model_input.side_effect = MMSError(err.INVALID_PREDICT_INPUT, 'Some message')

            input_batch, req_to_id_map, invalid_reqs = worker.retrieve_data_for_inference(requests=self.valid_req)

            worker.retrieve_model_input.assert_called_once()
            assert invalid_reqs == {'111-222-3333': err.INVALID_PREDICT_INPUT}

        def test_valid_req(self, patches, worker):
            worker.retrieve_model_input.return_value = 'some-return-val'
            worker.retrieve_data_for_inference(requests=self.valid_req)

            patches.validate.assert_called()
            worker.retrieve_model_input.assert_called()

    class TestPredict:

        data = {u'modelName': 'modelName', u'requestBatch': ['data']}

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['validate', 'emit', 'model_service'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators.validate_predict_msg'),
                mocker.patch('mms.model_service_worker.emit_metrics'),
                Mock(['metrics_init', 'inference', 'metrics_store'])
            )
            patches.model_service.metrics_store.store = Mock()
            return patches

        @pytest.fixture()
        def worker(self, mocker, patches):
            mocker.patch.object(MXNetModelServiceWorker, 'service_manager', create=True),
            mocker.patch.object(MXNetModelServiceWorker, 'retrieve_data_for_inference'),
            mocker.patch.object(MXNetModelServiceWorker, 'create_predict_response'),
            worker = object.__new__(MXNetModelServiceWorker)
            worker.service_manager.get_loaded_modelservices.return_value = {'modelName': patches.model_service}
            worker.retrieve_data_for_inference.return_value = [{0: 'inputBatch1'}], 'req_id_map', 'invalid_reqs'
            return worker

        def test_value_error(self, patches, worker):
            patches.validate.side_effect = ValueError('testerr')
            with pytest.raises(MMSError) as excinfo:
                worker.predict(self.data)
            assert excinfo.value.code == err.INVALID_PREDICT_MESSAGE
            assert excinfo.value.message == "ValueError('testerr',)"

        def test_pass_mms_error(self, patches, worker):
            error = MMSError(err.UNKNOWN_COMMAND, 'testerr')
            worker.service_manager.get_loaded_modelservices.side_effect = error
            with pytest.raises(MMSError) as excinfo:
                worker.predict(self.data)
            assert excinfo.value == error

        def test_not_loaded(self, patches, worker):
            worker.service_manager.get_loaded_modelservices.return_value = []
            with pytest.raises(MMSError) as excinfo:
                worker.predict(self.data)
            assert excinfo.value.code == err.MODEL_SERVICE_NOT_LOADED
            assert excinfo.value.message == "Model modelName is currently not loaded"

        def test_invalid_batch_size(self, patches, worker):
            data = {u'modelName': 'modelName', u'requestBatch': ['data1', 'data2']}
            with pytest.raises(MMSError) as excinfo:
                worker.predict(data)
            assert excinfo.value.code == err.UNSUPPORTED_PREDICT_OPERATION
            assert excinfo.value.message == "Invalid batch size 2"

        def test_success(self, patches, worker):
            response, msg, code = worker.predict(self.data)
            patches.validate.assert_called_once_with(self.data)
            worker.retrieve_data_for_inference.assert_called_once_with(['data'])
            patches.model_service.inference.assert_called_once_with(['inputBatch1'])
            patches.emit.assert_called_once_with(patches.model_service.metrics_store.store)
            worker.create_predict_response.assert_called_once_with([patches.model_service.inference()], 'req_id_map',
                                                                   'invalid_reqs')
            assert response == worker.create_predict_response()
            assert msg == "Prediction success"
            assert code == 200

    class TestLoadModel:

        data = {'modelPath': 'mpath', 'modelName': 'name', 'handler': 'handled'}

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['validate', 'loader'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators.validate_load_message'),
                mocker.patch('mms.model_service_worker.ModelLoader.load')
            )
            patches.loader.return_value = 'testmanifest', 'test_service_file_path'
            return patches

        @pytest.fixture()
        def worker(self, mocker):
            mocker.patch.object(MXNetModelServiceWorker, 'service_manager', create=True),
            return object.__new__(MXNetModelServiceWorker)

        def test_load_value_error(self, patches, worker):
            patches.loader.side_effect = ValueError('verror')
            with pytest.raises(MMSError) as excinfo:
                worker.load_model(self.data)
            assert excinfo.value.code == err.VALUE_ERROR_WHILE_LOADING
            assert excinfo.value.message == 'verror'

        def test_pass_mms_error(self, patches, worker):
            error = MMSError(err.UNKNOWN_COMMAND, 'testerr')
            patches.loader.side_effect = error
            with pytest.raises(MMSError) as excinfo:
                worker.load_model(self.data)
            assert excinfo.value == error

        def test_unknown_error(self, patches, worker):
            patches.loader.side_effect = Exception('testerr')
            with pytest.raises(MMSError) as excinfo:
                worker.load_model(self.data)
            assert excinfo.value.code == err.UNKNOWN_EXCEPTION_WHILE_LOADING
            assert excinfo.value.args[0] == "Exception('testerr',)"

        @pytest.mark.parametrize('batch_size', [(None, None), ('1', 1)])
        @pytest.mark.parametrize('gpu', [(None, None), ('2', 2)])
        def test_optional_args(self, patches, worker, batch_size, gpu):
            data = self.data.copy()
            if batch_size[0]:
                data['batchSize'] = batch_size[0]
            if gpu[0]:
                data['gpu'] = gpu[0]
            worker.load_model(data)
            worker.service_manager.register_and_load_modules.assert_called_once_with('name', 'mpath', 'testmanifest',
                                                                                     'test_service_file_path', gpu[1],
                                                                                     batch_size[1])

        def test_success(self, patches, worker):
            msg, code = worker.load_model(self.data)
            patches.validate.assert_called_once_with(self.data)
            patches.loader.assert_called_once_with('mpath', 'handled')
            assert msg == 'loaded model test_service_file_path'
            assert code == 200

    class TestUnloadModel:

        request = {u'model-name': 'mname'}

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['validate'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators.validate_unload_msg')
            )
            return patches

        @pytest.fixture()
        def worker(self, mocker):
            mocker.patch.object(MXNetModelServiceWorker, 'service_manager', create=True),
            return object.__new__(MXNetModelServiceWorker)

        def test_not_loaded(self, patches, worker):
            worker.service_manager.unload_models.side_effect = KeyError()
            with pytest.raises(MMSError) as excinfo:
                worker.unload_model(self.request)
            assert excinfo.value.code == err.MODEL_CURRENTLY_NOT_LOADED
            assert excinfo.value.args[0] == 'Model is not being served on model server'

        def test_pass_mms_error(self, patches, worker):
            error = MMSError(err.UNKNOWN_COMMAND, 'testerr')
            worker.service_manager.unload_models.side_effect = error
            with pytest.raises(MMSError) as excinfo:
                worker.unload_model(self.request)
            assert excinfo.value == error

        def test_unknown_error(self, patches, worker):
            worker.service_manager.unload_models.side_effect = Exception('testerr')
            with pytest.raises(MMSError) as excinfo:
                worker.unload_model(self.request)
            assert excinfo.value.code == err.UNKNOWN_EXCEPTION
            assert excinfo.value.args[0] == "Unknown error Exception('testerr',)"

        def test_success(self, patches, worker):
            msg, code = worker.unload_model(self.request)
            patches.validate.assert_called_once_with(self.request)
            worker.service_manager.unload_models.assert_called_once_with('mname')
            assert msg == "Unloaded model mname"
            assert code == 200

    class TestStopServer:

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['log'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.log_msg'),
            )
            return patches

        @pytest.fixture()
        def sock(self):
            return Mock()

        @pytest.fixture()
        def worker(self, mocker):
            mocker.patch.object(MXNetModelServiceWorker, 'send_response')
            return object.__new__(MXNetModelServiceWorker)

        def test_with_nil_sock(self, worker):
            with pytest.raises(ValueError) as excinfo:
                worker.stop_server(None)

            assert isinstance(excinfo.value, ValueError)
            assert excinfo.value.args[0] == "Invalid parameter passed to stop server connection"

        def test_with_exception(self, patches, worker, sock):
            close_exception = Exception("exception")
            sock.close.side_effect = close_exception
            log_call_param = "Error closing the socket {}. Msg: {}".format(sock, repr(close_exception))

            worker.stop_server(sock)

            sock.close.assert_called()
            patches.log.assert_called_with(log_call_param)

        def test_stop_server(self, worker, sock):
            worker.stop_server(sock)
            worker.send_response.assert_called()
            sock.close.assert_called()


def test_emit_metrics(mocker, socket_patches):
    metrics = {'test_emit_metrics': True}
    dumps = mocker.patch('json.dumps')
    emit_metrics(metrics)
    socket_patches.log_msg.assert_called()

    for k in metrics.keys():
        assert k in dumps.call_args_list[0][0][0]


class TestHandleConnection:

    @pytest.fixture()
    def get_spies(self, model_service_worker, mocker):
        Patches = namedtuple('Patches', ['recv_msg', 'predict', 'stop_server', 'load_model', 'unload_model',
                                         'create_and_send_response'])
        mock_patch = Patches(mocker.patch.object(model_service_worker, 'recv_msg', wraps=model_service_worker.recv_msg),
                             mocker.patch.object(model_service_worker, 'predict', wraps=model_service_worker.predict),
                             mocker.patch.object(model_service_worker, 'stop_server',
                                                 wraps=model_service_worker.stop_server),
                             mocker.patch.object(model_service_worker, 'load_model',
                                                 wraps=model_service_worker.load_model),
                             mocker.patch.object(model_service_worker, 'unload_model',
                                                 wraps=model_service_worker.unload_model),
                             mocker.patch.object(model_service_worker, 'create_and_send_response',
                                                 wraps=model_service_worker.create_and_send_response))

        return mock_patch

    def test_with_exit(self, model_service_worker, socket_patches, get_spies):
        get_spies.recv_msg.return_value = 'SToP', 'somedata'  # Since we do a cmd.lower() so testing with a variant...

        with pytest.raises(SystemExit) as ex:
            model_service_worker.handle_connection(socket_patches.socket)

        get_spies.recv_msg.assert_called()
        get_spies.stop_server.assert_called()
        assert ex.value.args[0] == 1

    def test_with_predict(self, model_service_worker, socket_patches, get_spies):
        get_spies.recv_msg.side_effect = [('PreDiCT', 'somedata'), ('stop', 'somedata')]

        with pytest.raises(SystemExit):
            model_service_worker.handle_connection(socket_patches.socket)

        get_spies.predict.assert_called()
        get_spies.recv_msg.assert_called()
        get_spies.stop_server.assert_called()

    def test_with_load(self, model_service_worker, socket_patches, get_spies):
        get_spies.recv_msg.side_effect = [('load', 'somedata'), ('stop', 'somedata')]

        with pytest.raises(SystemExit):
            model_service_worker.handle_connection(socket_patches.socket)

        get_spies.load_model.assert_called()
        get_spies.recv_msg.assert_called()
        get_spies.stop_server.assert_called()

    def test_with_unload(self, model_service_worker, socket_patches, get_spies):
        get_spies.recv_msg.side_effect = [('unload', 'somedata'), ('stop', 'somedata')]

        with pytest.raises(SystemExit):
            model_service_worker.handle_connection(socket_patches.socket)

        get_spies.unload_model.assert_called()
        get_spies.recv_msg.assert_called()
        get_spies.stop_server.assert_called()

    def test_with_unknown_cmd(self, model_service_worker, socket_patches, get_spies):
        result = "Received unknown command: {}".format('unk')
        err_code = err.UNKNOWN_COMMAND

        get_spies.recv_msg.side_effect = [('unk', 'somedata'), ('stop', 'somedata')]

        with pytest.raises(SystemExit):
            model_service_worker.handle_connection(socket_patches.socket)

        get_spies.create_and_send_response.assert_called_with(socket_patches.socket, err_code, result, None)
        get_spies.recv_msg.assert_called()
        get_spies.stop_server.assert_called()

    def test_with_mms_error(self, model_service_worker, socket_patches, get_spies):
        error = MMSError(err.SEND_FAILS_EXCEEDS_LIMITS, "Unknown Error")
        get_spies.recv_msg.side_effect = error

        model_service_worker.handle_connection(socket_patches.socket)
        socket_patches.log_error.assert_called()

    def test_with_mms_unknown_error(self, model_service_worker, socket_patches, get_spies):
        error = Exception("Unknown Error")
        get_spies.recv_msg.side_effect = [error, ('stop', 'somedata')]

        with pytest.raises(SystemExit):
            model_service_worker.handle_connection(socket_patches.socket)

        socket_patches.log_error.assert_called()
        get_spies.create_and_send_response.assert_called()
