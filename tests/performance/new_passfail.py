from bzt.modules.passfail import FailCriterion, PassFailStatus
from abc import abstractmethod


class FailCriterion1(object):
    def __init__(self, config, owner):
        self.owner = owner
        self.config = config
        self.agg_buffer = OrderedDict()
        if not 'threshold' in config:
            raise TaurusConfigError("Criteria string is malformed in its threshold part.")
        self.percentage = str(config['threshold']).endswith('%')
        if not 'subject' in config:
            raise TaurusConfigError("Criteria string is malformed in its subject part.")
        if config['subject'] == 'bytes':
            self.threshold = get_bytes_count(config.get('threshold'))
        else:
            self.threshold = dehumanize_time(config.get('threshold'))

        self.get_value = self._get_field_functor(config['subject'], self.percentage)
        self.window_logic = config.get('logic', 'for')
        self.agg_logic = self._get_aggregator_functor(self.window_logic, config['subject'])
        if not 'condition' in config:
            raise TaurusConfigError("Criteria string is malformed in its condition part.")
        self.condition = self._get_condition_functor(config.get('condition'))

        self.stop = config.get('stop', True)
        self.fail = config.get('fail', True)
        self.message = config.get('message', None)
        self.window = dehumanize_time(config.get('timeframe', 0))
        self._start = sys.maxsize
        self._end = 0
        self.is_candidate = False
        self.is_triggered = False

        self.get_threshold = config.get('get_threshold', None)
        if self.get_threshold:
            self.threshold == 'init'
            self.prev_data_str = None

    def __repr__(self):
        if self.is_triggered:
            if self.fail:
                state = "Failed"
            else:
                state = "Notice"
        else:
            state = "Alert"

        if self.message is not None:
            return "%s: %s" % (state, self.message)
        else:
            data = (state,
                    self.config['subject'],
                    self.config['condition'],
                    self.config['threshold'],
                    self.window_logic,
                    self.get_counting())
            return "%s: %s%s%s %s %d sec" % data

    def process_criteria_logic(self, tstmp, get_value):
        value = self.agg_logic(tstmp, get_value)
        if self.threshold == 'init':
            self.threshold = value
        state = self.condition(value, self.threshold)

        if self.window_logic == 'for':
            if state:
                self._start = min(self._start, tstmp)
                self._end = tstmp
            else:
                self._start = sys.maxsize
                self._end = 0

            if self.get_counting() >= self.window:
                self.trigger(value)
        elif self.window_logic == 'within' and state:
            self._start = tstmp - self.window + 1
            self._end = tstmp
            self.trigger(value)
        elif self.window_logic == 'over' and state:
            min_buffer_tstmp = min(self.agg_buffer.keys())
            self._start = min_buffer_tstmp
            self._end = tstmp
            if self.get_counting() >= self.window:
                self.trigger(value)

        self.owner.log.debug("%s %s: %s", tstmp, self, state)

    def trigger(self, value):
        print("triggered value is "+str(value))
        if self.get_threshold:
            self.threshold = value
            data_search = [self.config['subject'], self.config['condition'],
                           self.config['threshold'], self.window_logic]
            data_search_str = ','.join(data_search)
            data_str = ','.join(data_search + [value])
            replaced = False

            with open(self.get_threshold, "r+") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.startswith(data_search_str):
                        lines[i] = data_str
                        replaced = True
                        break

                if not replaced:
                    lines.append(data_str)

                f.seek(0)
                f.writelines(lines)

        else:
            if not self.is_triggered:
                self.owner.log.warning("%s", self)
            self.is_triggered = True

    def check(self):
        """
        Interrupt the execution if desired condition occured
        :raise AutomatedShutdown:
        """
        if self.stop and self.is_triggered:
            if self.fail:
                self.owner.log.info("Pass/Fail criterion triggered shutdown: %s", self)
                raise AutomatedShutdown("%s" % self)
            else:
                return True
        return False

    @abstractmethod
    def _get_field_functor(self, subject, percentage):
        pass

    def _get_condition_functor(self, cond):
        if cond == '=' or cond == '==':
            return lambda x, y: x == y
        elif cond == '>':
            return lambda x, y: x > y
        elif cond == '>=':
            return lambda x, y: x >= y
        elif cond == '<':
            return lambda x, y: x < y
        elif cond == '<=':
            return lambda x, y: x <= y
        else:
            raise TaurusConfigError("Unsupported fail criteria condition: %s" % cond)

    def _get_aggregator_functor(self, logic, _subject):
        if logic == 'for':
            return lambda tstmp, value: value
        elif logic in ('within', 'over'):
            return self._within_aggregator_avg  # FIXME: having simple average for percented values is a bit wrong
        else:
            raise TaurusConfigError("Unsupported window logic: %s" % logic)

    def _get_windowed_points(self, tstmp, value):
        self.agg_buffer[tstmp] = value
        keys = list(self.agg_buffer.keys())
        for tstmp_old in keys:
            if tstmp_old <= tstmp - self.window:
                del self.agg_buffer[tstmp_old]
                continue
            break

        return viewvalues(self.agg_buffer)

    def _within_aggregator_sum(self, tstmp, value):
        return sum(self._get_windowed_points(tstmp, value))

    def _within_aggregator_avg(self, tstmp, value):
        points = self._get_windowed_points(tstmp, value)
        return sum(points) / len(points)

    def get_counting(self):
        return self._end - self._start + 1


FailCriterion = FailCriterion1
