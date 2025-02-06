import os
import sys
import traceback
import datetime
import re
import tempfile
import ssl
from tempfile import NamedTemporaryFile

try:
    from sklearn.externals import joblib
except:
    import joblib

from Mining.selfmodule.toolmodule.dataprep import add_printindent
# from Mining.selfmodule.binarymodule.modelinfo import month_mark

original_out = sys.stdout
original_err = sys.stderr
sys_tem = 'win'  # 暂未添加linux，但不影响程序执行，只是不能隐藏文件而已

# 重定向至变量
class privy_Autonomy(object):
    """
    自定义变量的write方法
    """
    def __init__(self):
        """
        init
        """
        self._buff = ""

    def write(self, out_stream):
        """ :param out_stream:
        :return: """
        self._buff += out_stream

    def flush(self):
        pass


# current = sys.stdout
# a = privy_Autonomy() # 会调用a的write方法, 和self._buff的内容拼接
# sys.stdout = a
# print('testxj')
# sys.stdout = current # 输出捕获的内容
# print(a._buff)


# 从重定向变量的输出中，筛选需要打印在屏幕上的内容并print
def privy_print_filter(x, substring):
    for i in x._buff.split('\n'):
        if substring.lower() in i.lower():
            print(i)


# 重定向至屏幕 + 文件
# 隐藏约定：如果某些信息不想在打印时加入头部的时间标记，那么只需字符串的尾部添加空格，不影响打印信息的外观
class privy_Logger(object):
    def __init__(self, filename=None):
        """
        filename: 将输出重定向至文件，若取值为None，则在类的内部创建临时文件
        :param filename:
        """
        self.terminalout = sys.stdout
        self.terminalerr = sys.stderr
        self.istemp = False
        if filename is None:
            self.istemp = True
            self.filename = NamedTemporaryFile(mode="w+", dir=r"./", delete=True)
        else:
            self.filename = filename
        self.mes = ''

    def write(self, message):
        time_now = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')
        time_null = ' ' * (len(time_now) + len('  '))  #  与此保持一致：time_now + '  ' + message
        if self.istemp:
            self.log = self.filename
        else:
            self.log = open(self.filename, "a")
        if len(message) > 0:
            if re.search('Traceback|Error|File "|Warn|Exception', message):
                if re.search('Traceback|Warn', message):
                    m = '\n\n' + message
                else:
                    m = message
                self.terminalerr.write(m)
            else:
                if message == '\n':
                    m = message
                else:
                    message = add_printindent(message, time_null)
                    if re.search(' $', message):
                        m = message  # m = '\n' + message
                    else:
                        message = re.sub(f"^{time_null}", '', message)
                        m = time_now + '  ' + message  # m = '\n' + time_now + '  ' + message
                self.terminalout.write(m)
            self.mes += m
            self.log.write(m)

    def close(self):
        if self.istemp:
            self.filename.close()

    def flush(self):
        pass


class privy_lotest():

    def __init__(self, dirdict):
        """
        在自定义类中保存（隐藏）日志
        :param dirdict: 字典，逐层深入的key相当于一层层文件夹，value为日志长字符串
        """
        self.dirdict = dirdict


def privy_frame_fun(lt, wd_list):
    """
    为privy_lotest的dirdict属性匹配多级的key
    :param lt: privy_lotest类实例
    :param wd_list: 多级key，相当于多层文件夹
                    生成多级的key， 如：lt.dirdict['eg']['traintest~202010~202011']['log']['log~20220301110618']
    :return:
    """
    frame = lt.dirdict
    fr = ''
    for i in wd_list:
        if i not in frame.keys():
            fr += f"['{i}']"
            print(f'创建 .dirdict{fr}')
            frame[i] = {}
        frame = frame[i]
    return lt


# """
#  隐藏/取消隐藏文件
# :param x: 文件名
# :param method: 取值hide为隐藏文件
#                取值unhide为取消隐藏（文件存在则取消隐藏，文件不存在则跳过，不会报错）
# :param sys_tem: 系统，目前仅针对windowns系统，如需linux系统，请填充
# :return:
# """
def privy_fileattr_fun(x, method='hide', sys_tem=sys_tem):

    if sys_tem == 'win':
        import win32con, win32api
        if method == 'hide':
                win32api.SetFileAttributes(x, win32con.FILE_ATTRIBUTE_HIDDEN)
        elif method == 'unhide':
            if os.path.exists(x):
                win32api.SetFileAttributes(x, win32con.FILE_ATTRIBUTE_NORMAL)
            else:
                pass  # print(f'{x}不存在，忽略unhide')


def privy_log_save(lo, lt, log_pkl, filekey, Info, if_predict=False):
    """
    将日志隐藏至自定义类保存文件中（pkl）
    :param lo:  privy_Logger类实例，用于捕捉屏幕输出
    :param lt:  privy_lotest类实例 ，将lo步骤的输出，保存至其dirdict的对应多层键下的value中
    :param log_pkl: 日志pkl文件名称
    :param filekey: 代表日志文件名的键
    :param Info: 单个模型信息（命名元组）
    :param if_predict: 是否预测阶段
    :return: None,结果保存至文件
    """
    lo.close()
    modelkey = Info.short_name  # 模型名称
    stepkey = re.sub('.*/', '', Info.model_wd_predict if if_predict else Info.model_wd_traintest)  # 阶段名称 训练测试/预测打分账期

    wd_list = [modelkey, stepkey, 'log', filekey]
    lt = privy_frame_fun(lt, wd_list)
    path = ''.join([f"['{i}']" for i in wd_list])

    logpath = f"lt.dirdict{path}"
    print(f"将日志存至：{logpath}")
    exec(f"{logpath} = lo.mes")
    print(f'保存日志文件：{log_pkl}')
    privy_fileattr_fun(log_pkl, 'unhide')
    joblib.dump(lt, log_pkl)
    privy_fileattr_fun(log_pkl)


def privy_log_write(Info, step, log_type, log_pkl):
    """
    统一的日志写入函数
    :param Info: 单个模型信息（命名元组）
    :param step: 报错训练测试'traintest'、预测'predict'两个阶段的日志
    :param log_type: 日志类型，txt或pkl
    :param log_pkl: 日志pkl文件名称
    :return: privy_Logger实例（及pkl时保存日志所需的其他变量）
    """
    timemark = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if log_type == "txt":
        if step == 'traintest':
            log_wd_train = Info.model_wd_traintest + '/log'
            if not os.path.isdir(log_wd_train):
                print(f'创建目录：{log_wd_train}')
                os.makedirs(log_wd_train)
            mark_traintest = re.sub('^.*traintest', '', Info.model_wd_traintest)  # month_mark(Info, 'traintest')
            log_train = f"{log_wd_train}/{Info.short_name}_traintest{mark_traintest}~{timemark}.txt"
            print(f"日志文件：{log_train}")
            return privy_Logger(log_train),

        elif step == 'predict':
            log_wd_predict = f"{Info.model_wd_predict}/log"
            if not os.path.isdir(log_wd_predict):
                print(f'创建目录：{log_wd_predict}')
                os.makedirs(log_wd_predict)
            log_predict = f"{log_wd_predict}/{Info.short_name}_forecast~{Info.month_predict}~{timemark}.txt"
            print(f"日志文件：{log_predict}")
            return privy_Logger(log_predict),

    elif log_type == 'pkl':
        if step == 'traintest':
            if not os.path.isdir(Info.model_wd_traintest):
                print(f'创建目录：{Info.model_wd_traintest}')
                os.makedirs(Info.model_wd_traintest)

            if not os.path.exists(log_pkl):
                print(f'创建并保存空日志文件{log_pkl}')
                joblib.dump(privy_lotest({}), log_pkl)
                privy_fileattr_fun(log_pkl)

            print(f'\n加载日志文件：{log_pkl}')
            lt = joblib.load(log_pkl)
            filekey = f"log~{timemark}"  # 日志的键值
            lo = privy_Logger()
            return lo, lt, log_pkl, filekey

        elif step == 'predict':
            if not os.path.isdir(Info.model_wd_predict):
                print(f'创建目录：{Info.model_wd_predict}')
                os.makedirs(Info.model_wd_predict)

            print(f'\n加载日志文件：{log_pkl}')  # 训练时已经创建过该文件
            lt = joblib.load(log_pkl)
            filekey = f"log~{timemark}"  # 日志的键值
            lo = privy_Logger()
            return lo, lt, log_pkl, filekey
