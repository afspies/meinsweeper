from meinsweeper import MSLogger
import contextlib
import io
import unittest
import logging


class TestLocalLogger(unittest.TestCase):
    logger = MSLogger()
    # Redirect logger to another handler so we can capture outputs
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)
    logger.python_logger.addHandler(stream_handler)

    def test_train_loss(self):
        # Capture logging stream "INFO" in a string buffer
        with io.StringIO() as buf:
            # run the tests
            self.stream_handler.stream = buf
            self.logger.log_loss(0.0001, mode='train')
            self.assertEqual(buf.getvalue(), '[[LOG_ACCURACY TRAIN]] Step: 0; Losses: Train: 0.0001\n')
            self.stream_handler.flush()

    def test_validation_loss(self):
        # Capture logging stream "INFO" in a string buffer
        with io.StringIO() as buf:
            # run the tests
            self.stream_handler.stream = buf
            self.logger.log_loss(0.1, mode='val')
            self.assertEqual(buf.getvalue(), '[[LOG_ACCURACY VAL]] Step: 0; Losses: Val: 0.1\n')
            self.stream_handler.flush()

class TestLocalNode(unittest.TestCase):
    def test_node(self):
        pass


if __name__ == "__main__":
    unittest.main()
