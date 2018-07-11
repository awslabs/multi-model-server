import sys
import os
import pytest
from mms.metrics import Metrics
from mms.model_service_worker import emit_metrics
def test_metrics(capsys):
    """
    Test if metric classes methods behave as expected
    Also checks global metric service methods
    """
    # Create a batch of request ids
    request_ids = {0 : 'abcd', 1 :"xyz", 2 : "qwerty", 3 : "hjshfj" }

    model_name = "dummy model"

    # Create a metrics objects
    metrics = Metrics(request_ids, model_name)

    # Counter tests
    metrics.addCounter('CorrectCounter', 1, 1)
    assert 'CorrectCounter' in metrics.metrics[model_name]['xyz']
    metrics.addCounter('CorrectCounter',1, 1)
    metrics.addCounter('CorrectCounter',1, 3)
    metrics.addCounter('CorrectCounter', 1)
    assert 'CorrectCounter' in metrics.metrics[model_name]['ALL']
    metrics.addCounter('CorrectCounter', 3)
    assert metrics.metrics[model_name]['xyz']['CorrectCounter'].value == 2
    assert metrics.metrics[model_name]['hjshfj']['CorrectCounter'].value == 1
    assert metrics.metrics[model_name]['ALL']['CorrectCounter'].value == 4
    # Check what is emitted is correct
    emit_metrics(metrics.metrics)
    out, err = capsys.readouterr()
    assert '"abcd":{}' in out

    # Adding other types of metrics
    # Check for time metric
    try:
        metrics.addTime('WrongTime', 20, 1, 'ns')
    except Exception as e:
        assert "the unit for a timed metric should be" in str(e)

    metrics.addTime('CorrectTime', 20, 2, 's')
    metrics.addTime('CorrectTime', 20, 0)
    assert metrics.metrics[model_name]['abcd']['CorrectTime'].value == 20
    assert metrics.metrics[model_name]['abcd']['CorrectTime'].unit == 'Milliseconds'
    assert metrics.metrics[model_name]['qwerty']['CorrectTime'].value == 20
    assert metrics.metrics[model_name]['qwerty']['CorrectTime'].unit == 'Seconds'
    # Size based metrics
    try:
        metrics.addSize('WrongSize', 20, 1, 'TB')
    except Exception as e:
        assert "The unit for size based metric is one of" in str(e)

    metrics.addSize('CorrectSize', 200, 0, 'GB')
    metrics.addSize('CorrectSize', 10, 2)
    assert metrics.metrics[model_name]['abcd']['CorrectSize'].value == 200
    assert metrics.metrics[model_name]['abcd']['CorrectSize'].unit == 'Gigabytes'
    assert metrics.metrics[model_name]['qwerty']['CorrectSize'].value == 10
    assert metrics.metrics[model_name]['qwerty']['CorrectSize'].unit == 'Megabytes'

    # Check a percentage metric
    metrics.addPercent('CorrectPercent', 20.0, 3)
    assert metrics.metrics[model_name]['hjshfj']['CorrectPercent'].unit == 'Percent'

    # Check a error metric
    metrics.addError('CorrectError', 'Wrong values')
    assert metrics.metrics[model_name]['ERROR']['CorrectError'].unit == 'Error'
