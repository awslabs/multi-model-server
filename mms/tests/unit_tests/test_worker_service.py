import os
from collections import namedtuple

import pytest
from mms.context import Context
from mms.mxnet_model_service_error import MMSError
from mms.protocol.otf_message_handler import OtfCodecHandler
from mms.service import Service
from mms.service import emit_metrics
from mms.utils.model_server_error_codes import ModelServerErrorCodes as Err
from mock import Mock


# noinspection PyClassHasNoInit
class TestService:
    model_name = 'testmodel'
    model_dir = os.path.abspath('mms/tests/unit_tests/test_utils/')
    manifest = "testmanifest"
    data = {u'modelName': b'modelName', u'requestBatch': [{"requestId": b"123", "modelInputs": "", "data": b""}]}

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['validate'])
        patches = Patches(
            mocker.patch('mms.service.ModelWorkerMessageValidators.validate_predict_data')
        )
        return patches

    # noinspection PyUnusedLocal
    @pytest.fixture()
    def service(self, mocker, patches):
        service = object.__new__(Service)
        service.legacy = False
        service._entry_point = mocker.MagicMock(return_value=['prediction'])
        service._context = Context(self.model_name, self.model_dir, self.manifest, 1, 0, '1.0')
        service.retrieve_data_for_inference.return_value = [{0: 'inputBatch1'}], [b'2323'], 'invalid_reqs'
        return service

    # noinspection PyUnusedLocal
    def test_success(self, patches, service):
        codec = OtfCodecHandler()
        response, msg, code = service.predict(self.data, codec)

        assert msg == "Prediction success"
        assert code == 200


# noinspection PyClassHasNoInit
class TestEmitMetrics:
    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['log_msg'])
        patches = Patches(mocker.patch('mms.service.log_msg'), )
        return patches

    # noinspection PyUnusedLocal
    def test_emit_metrics(self, mocker, patches):
        metrics = {'test_emit_metrics': True}
        emit_metrics(metrics)
        patches.log_msg.assert_called()


# noinspection PyClassHasNoInit
class TestRetrieveDataForInference:
    valid_req = [{'requestId': b'111-222-3333', 'modelInputs': '[{}]'}]

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

    # noinspection PyUnusedLocal
    def test_with_nil_request(self, patches, service):
        with pytest.raises(ValueError) as excinfo:
            service.retrieve_data_for_inference(requests=None)

        assert excinfo.value.args[0] == "Received invalid inputs"

    # noinspection PyUnusedLocal
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

