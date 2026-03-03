import logging
import traceback

class LoggerFactory:
    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

class ExceptionHandler:
    @staticmethod
    def handle_exception(exc):
        logging.error('An exception occurred: %s', exc)
        logging.error('Traceback: %s', traceback.format_exc())

