import os


def get_config_path():
    return os.getenv("EHR_EXTRACT_CONFIGS")


def get_data_path():
    return os.getenv("EHR_EXTRACT_DATA")
