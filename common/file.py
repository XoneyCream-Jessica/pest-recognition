import os

__resource_path__ = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'resource')
__log_path__ = os.path.join(__resource_path__, 'log')
__data_path__ = os.path.join(__resource_path__, 'data')
__models_path__ = os.path.join(__resource_path__, 'models')


def check_and_create_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_data_path():
    return __data_path__


def get_models_path():
    return __models_path__


def get_log_path():
    return __log_path__


def __init__():
    check_and_create_folder(__resource_path__)
    check_and_create_folder(__log_path__)
    check_and_create_folder(__data_path__)
    check_and_create_folder(__models_path__)


__init__()
