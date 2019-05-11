import logging
import os
import sys

import pytest

from mms.context import Context
from mms.service import Service
from mms.service import emit_metrics

logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)


# noinspection PyClassHasNoInit
class TestService:

    model_name = 'testmodel'
    model_dir = os.path.abspath('mms/tests/unit_tests/test_utils/')
    manifest = "testmanifest"
    data = [
        {"requestId": b"123", "parameters": [
            {"name": "xyz", "value": "abc", "contentType": "text/csv"}
        ], "data": b""}
    ]

    @pytest.fixture()
    def service(self, mocker):
        service = object.__new__(Service)
        service._entry_point = mocker.MagicMock(return_value=['prediction'])
        service._context = Context(self.model_name, self.model_dir, self.manifest, 1, 0, '1.0')
        return service

    def test_predict(self, service, mocker):
        create_predict_response = mocker.patch("mms.service.create_predict_response")
        service.predict(self.data)
        create_predict_response.assert_called()

    def test_with_nil_request(self, service):
        with pytest.raises(ValueError, match=r"Received invalid inputs"):
            service.retrieve_data_for_inference(None)

    def test_valid_req(self, service):
        headers, input_batch, req_to_id_map = service.retrieve_data_for_inference(self.data)
        assert headers[0].get_request_property("xyz").get("content-type") == "text/csv"
        assert input_batch[0] == {"xyz": "abc"}
        assert req_to_id_map == {0: "123"}


# noinspection PyClassHasNoInit
class TestEmitMetrics:

    def test_emit_metrics(self, caplog):
        metrics = {'test_emit_metrics': True}
        emit_metrics(metrics)
        assert "[METRICS]" in caplog.text
