import datetime
import functools
import logging
import pprint
import random
import shutil
import string
import sys

import six


# ==============================================================================
# =                                   print                                    =
# ==============================================================================

pp = pprint.pprint
fp = functools.partial(six.print_, flush=True)


# ==============================================================================
# =                                  logging                                   =
# ==============================================================================

_logger = logging.getLogger('pylib_logger')
_handler = logging.StreamHandler(sys.stdout)
_formtter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s %(filename)s (line %(lineno)d): ========== %(message)s ==========', datefmt='%Y/%m/%d %H:%M:%S')

_handler.setFormatter(_formtter)
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)

debug = _logger.debug
info = _logger.info
warning = _logger.warning
error = _logger.error
critical = _logger.critical
log = _logger.log


# ==============================================================================
# =                                   random                                   =
# ==============================================================================

def rand_str_n(n, char_set=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(char_set) for _ in range(n))

rand_digit_str_n = functools.partial(rand_str_n, char_set=string.digits)
rand_lower_str_n = functools.partial(rand_str_n, char_set=string.ascii_lowercase)
rand_upper_str_n = functools.partial(rand_str_n, char_set=string.ascii_uppercase)


# ==============================================================================
# =                                    file                                    =
# ==============================================================================

cp = shutil.copy2


# ==============================================================================
# =                                    date                                    =
# ==============================================================================
# Note: standard (default) date string format: 'yyyy-mm-dd'

def str_to_date(str_date):
    return datetime.datetime.strptime(str_date, "%Y-%m-%d")


def date_to_str(date):
    return date.strftime("%Y-%m-%d")

str_from_date = date_to_str


def remove_cross_bar(str_date):
    return str_date.replace('-', '')


def add_cross_bar(str_date):
    return (datetime.datetime.strptime(str_date, "%Y%m%d")).strftime("%Y-%m-%d")


# ==============================================================================
# =                                   import                                   =
# ==============================================================================

def import_from_file(path, name='config'):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
