"""Learning Logging

https://realpython.com/python-logging/


Logging levels:
- DEBUG
- INFO
- WARNING
- ERROR
- CRITICAL

"""

import logging


# root logger

def root_logger():
    logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                        format='%(process)d %(levelname)s %(asctime)s %(name)s %(message)s')

    # logging.basicConfig(level=logging.DEBUG)

    # not printed
    logging.debug('this is a debug message')
    logging.info('this is an info message')

    # printed to console
    logging.warning('this is a warning message')
    logging.error('this is an error message')
    logging.critical('this is a critical message')


    name = 'John'

    logging.error(f'{name} raised and error!!!')


    a = 5
    b = 0

    try:
        c = a / b
    except Exception as e:
        logging.exception('Exception occurred')


# Using Classes

# custom logger
logger = logging.getLogger(__name__)

# create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('logging_file.log')
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)

# create formaters
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# logs
logger.warning('this is a warning')
logger.error('this is an error')

logging










