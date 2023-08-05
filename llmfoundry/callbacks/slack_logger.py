import os
import logging

from composer import Callback, Logger, State
from composer.utils import MissingConditionalImportError, dist

log = logging.getLogger(__name__)


class SlackLogger(Callback):

    def __init__(self):
        try:
            import slack_sdk
            self.client = slack_sdk.WebClient()
            del slack_sdk
        except ImportError as e:
            raise MissingConditionalImportError('slack_logger', 'slack_sdk',
                                                None) from e
        self.slack_logging_api_key = os.environ.get('SLACK_LOGGING_API_KEY')
        self.channel_id = os.environ.get('SLACK_LOGGING_CHANNEL_ID', None)

    def fit_end(self, state: State, logger: Logger):
        if dist.get_global_rank() != 0:
            return
        message = f"QUICK THE TOKEN TRAIN IS LEAVING THE STATION\nRun finished: {state.run_name}"
        try:
            self.client.chat_postMessage(token=self.slack_logging_api_key,
                                         channel=self.channel_id,
                                         text=message)
        except Exception as e:
            log.error(f'Error logging to Slack: {e}')


if __name__ == "__main__":

    class TestState(object):
        run_name = 'test_run'

    test_state = TestState()
    slack_logger = SlackLogger()
    slack_logger.fit_end(test_state, None)