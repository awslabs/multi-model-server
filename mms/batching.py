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
Batching strategy definition implementations for manual, naive, and AIMD batching strategies
"""

from abc import ABCMeta, abstractmethod
import time

from mms.log import get_logger

logger = get_logger()


def get_batching_strategy(strategy_name, config):
    """
    Initializes and returns the appropriate batching strategy according to strategy_name with config
    """
    strategy_name = strategy_name.lower()
    try:
        if strategy_name == "manual":
            return ManualBatchingStrategy(**config)
        elif strategy_name == "aimd":
            return AIMDBatchingStrategy(**config)
        elif strategy_name == "naive":
            return NaiveBatchingStrategy(**config)
        else:
            raise Exception("%s is not a valid batching strategy" % strategy_name)
    except TypeError:
        raise Exception("%s is not a valid configuration for %s batching strategy." % (str(config), strategy_name))


class BatchingStrategy(object):
    """
    Base class for backend batching strategies
    """
    __metaclass__ = ABCMeta

    # TODO replace sleep_time w/ latency in naive strategy?
    def __init__(self, service_name, data_store, input_type, batch_size, sleep_time):
        self.service_name = service_name
        self.data_store = data_store
        self.input_type = input_type
        self.batch_size = batch_size
        self.sleep_time = sleep_time

    @abstractmethod
    def wait_for_batch(self):
        """
        Returns
        -------
        ids : list of request UUIDs
        data : list of request data
        """
        pass


class NaiveBatchingStrategy(BatchingStrategy):
    """
    NaiveBatchingStrategy polls the queue at a set interval (sleep_time) and returns requests up to batch_size
    """
    def __init__(self, service_name, data_store, input_type, batch_size, sleep_time, **kwargs):
        super(NaiveBatchingStrategy, self).__init__(service_name, data_store, input_type, batch_size, sleep_time)

    def wait_for_batch(self):
        while True:
            ids, data = self.data_store.pop_batch(self.service_name, self.batch_size, self.input_type)

            assert len(ids) == len(data)
            if ids:
                return ids, data

            time.sleep(self.sleep_time)


class ManualBatchingStrategy(BatchingStrategy):
    """
    ManualBatchingStrategy polls the queue at a set interval. Once there is a request in the queue,
    it returns when either the batch size or the latency is exceeded.
    """
    def __init__(self, service_name, data_store, input_type, batch_size, sleep_time, latency, **kwargs):
        super(ManualBatchingStrategy, self).__init__(service_name, data_store, input_type, batch_size, sleep_time)
        self.latency = latency

    def wait_for_batch(self):
        start_time, previous_length = -1, float('-inf')
        while True:
            length = self.data_store.queue_len(self.service_name)
            if length != 0:
                if start_time == -1:
                    start_time = time.time()
                else:
                    if length < previous_length:
                        # Another worker processed the batch
                        start_time, previous_length = -1, float('-inf')
                        continue

                    latency_hit = time.time() - start_time >= self.latency
                    batch_size_hit = length >= self.batch_size
                    if latency_hit or batch_size_hit:
                        ids, data = self.data_store.pop_batch(self.service_name, self.batch_size, self.input_type)

                        assert len(ids) == len(data)
                        if ids:
                            return ids, data
                        else:
                            start_time, previous_length = -1, float('-inf')

            time.sleep(self.sleep_time)


class AIMDBatchingStrategy(BatchingStrategy):
    """
    AIMDBatchingStrategy attempts to find the optimal batch size under the latency constraint by
    adding a set amount to the previous batch size until the latency is over the constraint.
    It then multiplicatively decreases the batch size by a set factor until under the constraint
    """
    def __init__(self, service_name, data_store, input_type,
                 batch_size, sleep_time, latency, starting_batch_size, increase_amount, decrease_factor, **kwargs):
        super(AIMDBatchingStrategy, self).__init__(service_name, data_store, input_type, batch_size, sleep_time)
        self.latency = latency

        self.batch_size = starting_batch_size
        self.max_batch_size = batch_size
        self.increase_amount = increase_amount
        self.decrease_factor = decrease_factor

    def wait_for_batch(self):
        start_time, previous_length = -1, float('-inf')
        ids, data, latency = None, None, None
        while True:
            length = self.data_store.queue_len(self.service_name)
            if length != 0:
                if start_time == -1:
                    start_time = time.time()
                else:
                    if length < previous_length:
                        # Another worker processed the batch
                        start_time, previous_length = -1, float('-inf')
                        continue

                    batch_size_hit = length >= self.batch_size
                    if batch_size_hit:
                        ids, data = self.data_store.pop_batch(self.service_name, self.batch_size, self.input_type)

                        assert len(ids) == len(data)
                        if ids:
                            latency = time.time() - start_time
                            break
                        else:
                            start_time, previous_length = -1, float('-inf')

            time.sleep(self.sleep_time)

        if latency < self.latency:
            if self.batch_size >= self.max_batch_size:
                logger.debug("Batch size set to maximum.")
            else:
                self.batch_size += self.increase_amount
        elif latency > self.latency:
            if self.batch_size == 1:
                logger.debug("Batch size set to 1.")
            else:
                self.batch_size = int(self.batch_size * self.decrease_factor)

        return ids, data
