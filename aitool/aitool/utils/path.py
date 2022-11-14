import os
import glob
import six


def get_basename(file_path):
    """get base file name of file or path (no postfix)

    Args:
        file_path (str): input path or file

    Returns:
        str: base name
    """
    basename = os.path.splitext(os.path.basename(file_path))[0]

    return basename

def get_file_list(path, postfix='.png'):
    """get specific file path from the input path

    Args:
        path (str): input path
        postfix (str, optional): the postfix of return file. Defaults to '.png'.

    Returns:
        list: full file paths
    """

    file_list = glob.glob(f"{path}/*{postfix}")

    return file_list

def get_basename_list(path, postfix='.png'):
    """get specific filename from the input path

    Args:
        path (str): input path
        postfix (str, optional): the postfix of return file. Defaults to '.png'.

    Returns:
        list: full file paths
    """

    basename_list = [get_basename(_) for _ in os.listdir(path)]

    return basename_list

def mkdir_or_exist(dir_name, mode=0o777):
    """make or judge the exist of dir

    Args:
        dir_name (str): input dir
        mode (int, optional): inout mode. Defaults to 0o777.
    """
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)

def get_dir_name(file_path):
    """get the dir name

    Args:
        file_path (str): input path of file

    Returns:
        str: dir name
    """
    dir_name = os.path.abspath(os.path.dirname(file_path))

    return dir_name