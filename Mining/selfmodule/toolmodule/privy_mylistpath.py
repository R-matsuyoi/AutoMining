import os
from pandas import Series


def print_files(path):
    lsdir = os.listdir(path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
    if dirs:
        for i in dirs:
            sub_dir = os.path.join(path, i)
            if not os.listdir(sub_dir):
                print(sub_dir)
            else:
                print_files(sub_dir)
    files = [i for i in lsdir if os.path.isfile(os.path.join(path,i))]
    for f in files:
        print(os.path.join(path, f))


def getAllFilesize(path):
    """
    :param path: 文件夹
    :return: 返回文件夹下所有文件并计算大小
    """
    dir_size = Series(dtype=float, name='size_B')
    path = path.replace('\\', '/')

    def getDirSize(path):  # 获取文件夹下所有文件的大小
        if path[-1].__eq__('/'):  # 文件夹结尾判断有没有'/'
            pass
        else:
            path = path + '/'
        global dirSize  # 全局变量
        fileList = os.listdir(path)  # 获得文件夹下面的所有内容
        for i in fileList:
            if os.path.isdir(path + i):  # 如果是文件夹  那就再次调用函数去递归
                getDirSize(path + i)
            else:
                size = os.path.getsize(path + i)  # 获取文件的大小
                dir_size.loc[path + i] = size

    getDirSize(path)
    dir_size.index.name = 'filename'
    dir_size = dir_size.reset_index()
    dir_size['dir'] = dir_size.filename.apply(lambda x: os.path.dirname(x))
    dir_size['filename'] = dir_size['filename'].str.replace('^.*/', '')
    dir_size['size_MB'] = round(dir_size.size_B/(1024**2), 2)
    totalsize = round(dir_size.size_B.sum() / (1024**2), 2)
    print(f"{path}：{totalsize}MB")
    return dir_size[['dir', 'filename', 'size_B', 'size_MB']]
