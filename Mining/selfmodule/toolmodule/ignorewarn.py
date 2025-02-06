# 提前通过 with warnings.catch_warnings(record=True) as w: 捕捉警告，然后用该函数过滤掉某些不想展示的警告

import warnings
from pandas import Series


def filter_warnings(w):
    if w == []:
        return None

    ignore_cases = ["Pandas doesn't allow columns to be created via a new attribute name",
                    'future version',
                    'The default value of',
                    'Specify dtype option on import or set low_memory=False',  # 亚信
                    'From version',
                    'verbose参数值被赋予print_indent后设置为None',
                    'future releases',
                    'restore the old behavior',
                    'will be removed in a future release'
                    ]
    messages = Series(w).apply(lambda x: str(x.message))
    if_pass = messages.str.contains('|'.join(ignore_cases))
    send_out = messages[~if_pass]
    if len(send_out) > 0:
        send_out.apply(warnings.warn)