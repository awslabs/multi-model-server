
import pytest
from mock import Mock
import os
from collections import namedtuple
from mms.service import Service
from mms.context import Context
from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as Err
from mms.service import emit_metrics


class TestService:

    model_name = 'testmodel'
    model_dir = os.path.abspath('mms/tests/unit_tests/test_utils/')
    manifest = "testmanifest"
    data = {u'modelName': 'modelName', u'requestBatch': ['data']}

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['validate'])
        patches = Patches(
            mocker.patch('mms.service.ModelWorkerMessageValidators.validate_predict_msg'),
        )
        return patches

    @pytest.fixture()
    def service(self, mocker, patches):
            mocker.patch.object(Service, 'retrieve_data_for_inference'),
            mocker.patch.object(Service, 'create_predict_response'),
            service = object.__new__(Service)
            service.legacy = False
            service._entry_point = mocker.MagicMock(return_value='prediction')
            service._context = Context(self.model_name, self.model_dir, self.manifest,1,0, '1.0')
            service.retrieve_data_for_inference.return_value = [{0: 'inputBatch1'}], ['2323'], 'invalid_reqs'
            return service

    def test_value_error(self, patches, service):
        patches.validate.side_effect = ValueError('testerr')
        with pytest.raises(MMSError) as excinfo:
            service.predict(self.data)
        assert excinfo.value.code == Err.INVALID_PREDICT_MESSAGE
        assert excinfo.value.message == "ValueError('testerr',)"

    def test_success(self, patches, service):
        response, msg, code = service.predict(self.data)
        patches.validate.assert_called_once_with(self.data)
        service.retrieve_data_for_inference.assert_called_once_with(['data'])
        service._entry_point.assert_called_once_with(service.context, ['inputBatch1'])
        service.create_predict_response.assert_called_once_with(["prediction"], ['2323'],
                                                               'invalid_reqs')
        assert response == service.create_predict_response()
        assert msg == "Prediction success"
        assert code == 200

class TestEmitMetrics:
    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches',
                             ['log_msg'])
        patches = Patches(mocker.patch('mms.service.log_msg'),)
        return patches

    def test_emit_metrics(self, mocker, patches):
        metrics = {'test_emit_metrics': True}
        dumps = mocker.patch('json.dumps')
        emit_metrics(metrics)
        patches.log_msg.assert_called()

        for k in metrics.keys():
            assert k in dumps.call_args_list[0][0][0]

class TestRetrieveDataForInference:

    valid_req = [{'requestId': '111-222-3333', 'encoding': 'None|base64|utf-8', 'modelInputs': '[{}]'}]

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['validate'])
        patches = Patches(
            mocker.patch('mms.service.ModelWorkerMessageValidators.validate_predict_data')
        )
        return patches

    @pytest.fixture()
    def service(self, mocker):
        mocker.patch.object(Service, 'retrieve_model_input')
        return object.__new__(Service)

    def test_with_nil_request(self, patches, service):
        with pytest.raises(ValueError) as excinfo:
            service.retrieve_data_for_inference(requests=None)

        assert excinfo.value.args[0] == "Received invalid inputs"

    def test_with_invalid_req(self, patches, service):
        service.retrieve_model_input.side_effect = MMSError(Err.INVALID_PREDICT_INPUT, 'Some message')

        input_batch, req_to_id_map, invalid_reqs = service.retrieve_data_for_inference(requests=self.valid_req)

        service.retrieve_model_input.assert_called_once()
        assert invalid_reqs == {'111-222-3333': Err.INVALID_PREDICT_INPUT}

    def test_valid_req(self, patches, service):
        service.retrieve_model_input.return_value = 'some-return-val'
        service.retrieve_data_for_inference(requests=self.valid_req)

        patches.validate.assert_called()
        service.retrieve_model_input.assert_called()

class TestCreatePredictResponse:

    sample_ret = ['val1']
    req_id_map = {0: 'reqId1'}
    empty_invalid_reqs = dict()
    invalid_reqs = {'reqId1': 'invalidCode1'}

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['encode'])
        patches = Patches(
            mocker.patch('mms.service.ModelWorkerCodecHelper.encode_msg')
        )
        return patches

    @pytest.fixture()
    def service(self, mocker):
        return object.__new__(Service)

    def test_codec_exception(self, patches, service):
        patches.encode.side_effect = Exception('testerr')
        with pytest.raises(MMSError) as excinfo:
            service.create_predict_response(self.sample_ret, self.req_id_map, self.empty_invalid_reqs)
        assert excinfo.value.code == Err.CODEC_FAIL
        assert excinfo.value.message == "codec failed Exception('testerr',)"

    @pytest.mark.parametrize('value', [(b'test', b'test'), ('test', b'test'), ({'test': True}, b'{"test": true}')])
    def test_value_types(self, patches, value, service):
        ret = [value[0]]
        resp = service.create_predict_response(ret, self.req_id_map, self.empty_invalid_reqs)
        patches.encode.assert_called_once_with('base64', value[1])
        assert set(resp[0].keys()) == {'requestId', 'code', 'value', 'encoding'}

    @pytest.mark.parametrize('invalid_reqs,requestId,code,value,encoding', [
    (dict(), 'reqId1', 200, 'encoded', 'base64'),
    ({'reqId1': 'invalidCode1'}, 'reqId1', 'invalidCode1', 'encoded', 'base64')
    ])
    def test_with_or_without_invalid_reqs(self, patches, invalid_reqs, requestId, code, value, encoding, service):
        patches.encode.return_value = 'encoded'
        resp = service.create_predict_response(self.sample_ret, self.req_id_map, invalid_reqs)
        assert resp == [{'requestId': requestId, 'code': code, 'value': value, 'encoding': encoding}]

class TestRetreiveModelInput:

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['codec_helper', 'msg_validator'])
        patches = Patches(mocker.patch('mms.service.ModelWorkerCodecHelper'),
                          mocker.patch('mms.service.ModelWorkerMessageValidators'),
                          )
        return patches

    @pytest.fixture()
    def service(self, mocker):
        return object.__new__(Service)

    def test_retrieve_model_input(self, patches, service):
        valid_inputs = [{'encoding': 'base64', 'value': 'val1', 'name': 'model_input_name'}]

        patches.codec_helper.decode_msg.return_value = "some_decoded_resp"

        expected_response = {'model_input_name': 'some_decoded_resp'}

        model_in = service.retrieve_model_input(valid_inputs)

        patches.msg_validator.validate_predict_inputs.assert_called()
        patches.codec_helper.decode_msg.assert_called()

        assert expected_response == model_in
