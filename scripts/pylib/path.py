import datetime
import glob as _glob
import inspect
import os
import shutil
import sys


def add_path(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def remove_file_or_folder(path):
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        print(f'Unable to delete: {path} is not a file, directory, or symbolic link!')

remove = remove_file_or_folder


def split(path):
    """Return directory, name, ext."""
    directory, name_ext = os.path.split(path)
    name, ext = os.path.splitext(name_ext)
    return directory, name, ext


def directory(path):
    return split(path)[0]


def name(path):
    return split(path)[1]


def ext(path):
    return split(path)[2]


def name_ext(path):
    return ''.join(split(path)[1:])


def change_ext(path, ext):
    if ext[0] == '.':
        ext = ext[1:]
    return os.path.splitext(path)[0] + '.' + ext


asbpath = os.path.abspath


join = os.path.join


def prefix(path, prefixes, sep='-'):
    prefixes = prefixes if isinstance(prefixes, (list, tuple)) else [prefixes]
    directory, name, ext = split(path)
    return join(directory, sep.join(prefixes) + sep + name + ext)


def suffix(path, suffixes, sep='-'):
    suffixes = suffixes if isinstance(suffixes, (list, tuple)) else [suffixes]
    directory, name, ext = split(path)
    return join(directory, name + sep + sep.join(suffixes) + ext)


def prefix_now(path, fmt="%Y-%m-%d-%H:%M:%S", sep='-'):
    return prefix(path, prefixes=datetime.datetime.now().strftime(fmt), sep=sep)


def suffix_now(path, fmt="%Y-%m-%d-%H:%M:%S", sep='-'):
    return suffix(path, suffixes=datetime.datetime.now().strftime(fmt), sep=sep)


def glob(directory, pats='*', recursive=False, full=True):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(directory, pat), recursive=recursive)
    if not full:
        matches = [m.replace(os.path.join(directory, ''), '', 1) for m in matches]
    return matches


isdir = os.path.isdir
