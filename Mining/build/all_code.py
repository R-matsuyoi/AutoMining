# ---------------------------- <editor-fold desc="汇总所有代码"> -------------------------------------------------------
import re
import os
import pandas as pd
from pandas import DataFrame

# import importlib
import math
import inspect
import warnings
import re
from Mining.selfmodule.toolmodule.datatrans import *


def privy_replace_default(str_input, default_values):
    """
    默认值替换函数（只针对代码中以 arg = value的形式在函数前设置的默认值，不包括函数构造体内的默认值）
    :param str_input: 代码字符串
    :param default_values: 默认值替换字段 键值为参数名，值为新的默认值
    :return: 替换后的代码字符串
    """
    str_ad = re.sub('#.*?\n', '\n', str_input)
    for k, v in default_values.items():
        if isinstance(v, str):
            str_input = re.sub(f"{k} *= *'.*?'", f"{k} = '{v}'", str_input)
        elif isinstance(v, (int, float)):
            semi = re.findall(f'{k} *= *.* *[;]', str_ad)
            if len(semi) > 1:
                raise Exception(f'发现多处赋值语句：{semi}')
            elif len(semi) == 1:
                semi = semi[0]
            elif len(semi) == 0:
                semi = ''

            crlf = [i for i in re.findall(f'{k} *= *.*\n', re.sub(semi, '', str_ad)) if ',' not in i]
            if ((len(semi) != 0) & (len(crlf) > 0)) | (len(crlf) > 1):
                raise Exception(f'发现多处赋值语句：{([semi] if semi else []) + crlf}')
            elif len(crlf) == 1:
                crlf = crlf[0]
            elif len(crlf) == 0:
                crlf = ''

            if semi:
                str_input = str_input.replace(semi, f'{k} = {v};')
            elif crlf:
                str_input = str_input.replace(crlf, f'{k} = {v}\n')
            else:
                warnings.warn(f'未识别到 {k} 的赋值语句，请确认！')
    return str_input


def privy_upload_code(selfwd_platform, default_values=None, selfwd_source=None, per=50000, py_skip=None):
    """
    将本机中的所有代码上传至建模平台
    :param selfwd_platform: 平台上的个人目录
    :param default_values: privy_replace_default函数参数
    :param selfwd_source: 个人本机代码目录
    :param per: 若代码字符串过长则拆分， per为每份细分字符串的最大长度
    :return:
    """
    from Mining.selfmodule.toolmodule.privy_outredirect import privy_fileattr_fun
    from Mining.selfmodule.toolmodule.privy_mylistpath import getAllFilesize
    from Mining.selfmodule.toolmodule.datatrans import sql_format

    if selfwd_source is not None:
        print(f'将默认目录修改为selfwd_source（{selfwd_source}）')
        os.chdir(selfwd_source)

    def get_code(f):
        # if len(p.strip(' ')) == 0:
        #     return ''
        cut_off = '\n' + '#' * 100 + '\n'
        # f = path_fun(p)
        lines = open(f, encoding='UTF-8').read()
        for i in lines.split('\n'):
            i_ad = i.strip(' ').replace('  ', ' ')
            if i_ad.startswith('from selfmodule'):
                pass
            else:
                code_str.append(i.replace('\n', '\\n'))
        return ''

    code_str = [f'# {"$" * 3} 开始']
    code_str.append(re.sub('\n *', '\n', f"""
    import pandas as pd
    pd.options.display.max_columns = 30
    pd.options.display.max_rows = 500
    pd.set_option('display.width', 100000)
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import os
    original_wd = os.getcwd()
    """))

    pys = getAllFilesize('./Mining/selfmodule/')  # 获取所有模块文件
    skip = ['__init__.py'] + (list(py_skip) if py_skip else [])
    pys = pys.loc[pys.filename.str.endswith('.py') & (~pys.filename.isin(skip))]
    pys = pys.dir + '/' + pys.filename
    # 调整代码顺序：datatrans中包括其他函数需要用到的数据库等默认值设置
    pys = pd.concat([
        pys[pys.str.contains('datatrans.py')],
        pys[~pys.str.contains('datatrans.py')]])
    for f in pys:
        get_code(f)
    code_str.append(f'# {"$" * 3} 结束')
    code = '\n'.join(code_str)
    code = re.sub('__all__ *= *\[.*?]', '', code, flags=re.DOTALL)  # 去掉 __all__，否则只导入其中的对象
    if default_values is not None:
        code = privy_replace_default(code, default_values)

    code_init = code

    com = DataFrame({'co': re.findall(' #.*|^#.*|\n#.*', code)})
    com['le'] = com.co.apply(len)
    for i in com.sort_values(by='le', ascending=False).co:
        code = code.replace(i, '')

    code = code.replace('\n', '【《》】')
    com2 = [i for i in re.findall('""".*?"""', code) if ':param' in i]
    for i in com2:
        code = code.replace(i, '')

    while re.findall('【《》】 *【《》】 *【《》】', code):
        code = re.sub('【《》】 *【《》】 *【《》】', '【《》】【《》】', code)

    round = range(math.ceil(len(code) / per))
    s = 'code_add = ""'
    exec(s, locals())
    for i in round:
        start, end = per * i, per * (i + 1)
        s += '\n' + f'code_add += {repr(code[start: end])}'
        exec(s, locals())

    xj = sql_format("""
    if code_add != code:
        raise Exception(f'code_add（{len(code_add)}）的拼接结果与code（{len(code)}）不同，请检查！')""")
    exec(xj)

    s += '\n\n' + inspect.getsource(privy_fileattr_fun).replace('"""', "'''").replace('sys_tem=sys_tem', f"sys_tem='{default_values['sys_tem']}'")

    file_pkl = f'{selfwd_platform}/.pkl'.replace('//', '/')
    # 创建cltest类实例，code属性保存代码字符code_add，# 保存到.pkl
    s += sql_format(f"""
    import os
    try:
        from sklearn.externals import joblib
    except:
        import joblib
    class cltest():
        def __init__(self, code):
            self.code = code
    cl = cltest(code_add) 
    privy_fileattr_fun('{file_pkl}',  method='unhide')
    joblib.dump(cl, '{file_pkl}')
    privy_fileattr_fun('{file_pkl}')
    """)

    print('复制下列语句粘贴至建模平台并执行：将代码存至‘.pkl’中（首次一次性执行即可）\n')
    print(s)
    return code_init
# </editor-fold> -------------------------------------------------------------------------------------------------------


# -------------------------------------- <editor-fold desc="import所有代码"> -------------------------------------------
def privy_exec_code(plat, selfwd_platform, modelwd_platform, default_values=None, code_py_empty=True):
    """
    加载所有代码
    （为保证每个步骤都被执行，故务必以exec字符串的形式执行）
    :param plat: 平台
    :param selfwd_platform: 平台上的个人目录
    :param modelwd_platform: 平台上的模型目录
    :param default_values: privy_replace_default函数参数
    :param code_py_empty:
         True: 从‘.pkl’获取代码写入file_code，import所有代码，然后将file_code置空（代码完全隐藏，无痕执行）
         Flase:从‘.pkl’获取代码写入file_code（不得不对外展示代码，故不置空）
    :return:
    """
    from Mining.selfmodule.toolmodule.privy_outredirect import privy_fileattr_fun
    from Mining.selfmodule.toolmodule.datatrans import sql_format

    file_pkl = f'{selfwd_platform}/.pkl'.replace('//', '/')
    file_code = f'{selfwd_platform}/testcode.py'.replace('//', '/')
    # exe_str = '\n' + inspect.getsource(crypt).replace('"""', "'''")
    exe_str = '\n' + inspect.getsource(privy_fileattr_fun).replace('"""', "'''")
    exe_str += sql_format(f'''
    import os
    import sys
    try:
        from sklearn.externals import joblib
    except:
        import joblib
    import importlib
    #
    class cltest():
        def __init__(self, code):
            self.code = code
    code = joblib.load('{file_pkl}').code.replace('【《》】', '\\\\n')
    privy_fileattr_fun('{file_code}',  method='unhide')
    file_stream = open("{file_code}", "w", encoding="utf-8")
    file_stream.write(code)
    file_stream.flush()
    file_stream.close()
    ''')
    if code_py_empty:
        s_code_py_empty = sql_format(f"""
            #
            # 从文件中import代码
            sys.path.append('{selfwd_platform}')
            import testcode
            importlib.reload(testcode)
            from testcode import *
            #
            # import后将文件置空
            file_stream = open("{file_code}", "w", encoding="utf-8")
            file_stream.write('')
            file_stream.flush()
            file_stream.close()
            privy_fileattr_fun('{file_code}')
            #
            # 设置工作目录
            selfwd_platform = '{selfwd_platform}'
            original_wd = os.getcwd()
            print('默认目录原来为：',os.getcwd())
            os.chdir('{modelwd_platform}')
            print('默认目录修改为：', os.getcwd())
            log_pkl = f'{selfwd_platform}/ .pkl'
            log_type = 'pkl'
            plat = '{plat}'""")
        exe_str += s_code_py_empty

    if default_values is not None:
        exe_str = exe_str.replace('sys_tem=sys_tem', f"sys_tem='{default_values['sys_tem']}'")

    if code_py_empty:
        print('\n\n\n复制下列语句粘贴至建模平台并执行：相当于import所有代码的效果（每次都需执行）')
    else:
        print(f'\n\n\n复制下列语句粘贴至建模平台并执行：将代码写入{file_code}（首次一次性执行即可）')
    print(f'\nexec("""{exe_str}""")')

    if not code_py_empty:
        s2 = sql_format(f"""
            import sys
            import os
            selfwd_platform = '{selfwd_platform}'
            sys.path.append(selfwd_platform)
            from testcode import *
            original_wd = os.getcwd()
            print('默认目录原来为：',os.getcwd())
            os.chdir('{modelwd_platform}')
            print('默认目录修改为：', os.getcwd())
            log_pkl = f'{selfwd_platform}/ .pkl'
            log_type = 'pkl'
            plat = '{plat}'""")
        print('\n\n\n复制下列语句粘贴至建模平台并执行：import所有代码（每次都需执行）\n')
        print(s2)
# </editor-fold> -------------------------------------------------------------------------------------------------------
