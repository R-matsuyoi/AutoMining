from Mining.selfmodule.toolmodule.strtotable import string_to_table
from Mining.selfmodule.toolmodule.datatrans import *
from Mining.selfmodule.tablemodule.tablefun import month_add
from Mining.selfmodule.tablemodule.basestr import *

col_month_field = 'month_id'
col_month_partition = 'acct_month'


# <editor-fold desc="移网基础宽表_用户to套餐组 -----------------------------------------------------------------------">
def team_summary_sql(month):
    """
    从用户粒度生成套餐组粒度的汇总sql
    :param month: 处理账期
    :return: sql字符串
              python能直连数据库的环境在python中直接执行；
              python不能直连数据库的环境复制这段sql到数据库中手动执行
    """

    def fun(x):
        suffix_dict = {'_组内主卡取值': '_teamzk',
                       '_组内最大值': '_teammax',
                       '_组内总和': '_teamadd',
                       '': ''}
        col_new = x.原始字段名 + suffix_dict[x.汇总方式]
        if x.汇总方式 == '_组内主卡取值':
            sql = f"max(case when IS_DUPETY='0' then {x.原始字段名} else NULL end) {col_new}"
        elif x.汇总方式 == '_组内最大值':
            sql = f"max(case when IS_DUPETY='0' then {x.原始字段名} else NULL end) {col_new}"
        elif x.汇总方式 == '_组内总和':
            sql = f"sum(coalesce({x.原始字段名}, 0)) {col_new}"
        else:
            raise Exception(f'{x.汇总方式} 未实现，请修改代码！')
        return sql

    month = sql_value(month)
    field_team_method = string_to_table(s_yw_toteam)
    field_team_method = field_team_method[['原始字段名', '汇总方式']]
    field_team_method = field_team_method.apply(lambda c: c.str.strip(' '), axis=1)
    sql_col = ',\n'.join(field_team_method[field_team_method.汇总方式.notnull()].apply(fun, axis=1))

    # truncate table {prefix}dm_zc_yw_moxing_team_info_m partition ({col_month_partition}={month});
    sql = sql_format(f""" 
        insert overwrite table {prefix}dm_zc_yw_moxing_team_info_m partition ({col_month_partition}={month})
        select {col_month_field}, zk_user_no,
        {sql_col}
        from {prefix}dm_zc_yw_moxing_info_m where {col_month_partition}={month} -- and is_valid='1'
        group by {col_month_field}, zk_user_no
        """)
    sql = sql.lower().replace('5g_flux', 'flux_5g').replace('4g_flux', 'flux_4g').replace('3g_flux', 'flux_3g').replace(
        '2g_flux', 'flux_2g')
    return sql


# exam[exam.index.str.contains('count')].sort_values(by='stdad', ascending=False).head(50)
# exam[exam.index.str.contains('Q')].round(2).sort_values(by='stdad', ascending=False).head(50)
# </editor-fold-------------------------------------------------------------------------------------------------------->


#
#
# <editor-fold desc="移网补充特征表_套餐组：暂未加入宽表中的临时新增字段（原始） + 手动衍生_sql字段 ------------------">
def ywteam_xadd_sql(month, prefix=prefix):
    month = sql_value(month)

    # truncate table {prefix}dm_zc_yw_moxing_team_add_m partition ({col_month_partition}={month});
    sql = sql_format(f"""
        insert overwrite table {prefix}dm_zc_yw_moxing_team_add_m partition ({col_month_partition}={month}) 
        select 
         a.{col_month_field}
        ,a.zk_user_no
        ,sig_offer_spec_id_teamzk_count
        ,spec_count0_fk_num_teamzk
        ,spec_count1_fk_num_teamzk
        ,spec_count2_fk_num_teamzk
        ,spec_pre025_fk_num_teamzk
        ,spec_pre05_fk_num_teamzk
        ,spec_max_fk_num_teamzk
        ,spec_count0_comp_wf_num_teamzk
        ,spec_count1_comp_wf_num_teamzk
        ,spec_count2_comp_wf_num_teamzk
        ,spec_pre025_comp_wf_num_teamzk
        ,spec_pre05_comp_wf_num_teamzk
        ,spec_max_comp_wf_num_teamzk
        ,spec_count0_comp_kj_num_teamzk
        ,spec_count1_comp_kj_num_teamzk
        ,spec_count2_comp_kj_num_teamzk
        ,spec_pre025_comp_kj_num_teamzk
        ,spec_pre05_comp_kj_num_teamzk
        ,spec_max_comp_kj_num_teamzk
        from (select {col_month_field}, zk_user_no, sig_offer_spec_id_teamzk 
              from {prefix}dm_zc_yw_moxing_team_info_m where {col_month_partition}={month}
              ) a
        left join (
            select 
            sig_offer_spec_id_teamzk
            ,count(1) sig_offer_spec_id_teamzk_count
            ,sum(case when fk_num_teamzk > 0 then 1 else 0 end) spec_count0_fk_num_teamzk
            ,sum(case when fk_num_teamzk > 1 then 1 else 0 end) spec_count1_fk_num_teamzk
            ,sum(case when fk_num_teamzk > 2 then 1 else 0 end) spec_count2_fk_num_teamzk
            ,percentile_approx(fk_num_teamzk, 0.25) spec_pre025_fk_num_teamzk
            ,percentile_approx(fk_num_teamzk, 0.5) spec_pre05_fk_num_teamzk
            ,max(fk_num_teamzk) spec_max_fk_num_teamzk
            ,sum(case when comp_wf_num_teamzk > 0 then 1 else 0 end) spec_count0_comp_wf_num_teamzk
            ,sum(case when comp_wf_num_teamzk > 1 then 1 else 0 end) spec_count1_comp_wf_num_teamzk
            ,sum(case when comp_wf_num_teamzk > 2 then 1 else 0 end) spec_count2_comp_wf_num_teamzk
            ,percentile_approx(comp_wf_num_teamzk, 0.5) spec_pre025_comp_wf_num_teamzk
            ,percentile_approx(comp_wf_num_teamzk, 0.5) spec_pre05_comp_wf_num_teamzk
            ,max(comp_wf_num_teamzk) spec_max_comp_wf_num_teamzk
            ,sum(case when comp_kj_num_teamzk > 0 then 1 else 0 end) spec_count0_comp_kj_num_teamzk
            ,sum(case when comp_kj_num_teamzk > 1 then 1 else 0 end) spec_count1_comp_kj_num_teamzk
            ,sum(case when comp_kj_num_teamzk > 2 then 1 else 0 end) spec_count2_comp_kj_num_teamzk
            ,percentile_approx(comp_kj_num_teamzk, 0.5) spec_pre025_comp_kj_num_teamzk
            ,percentile_approx(comp_kj_num_teamzk, 0.5) spec_pre05_comp_kj_num_teamzk
            ,max(comp_kj_num_teamzk) spec_max_comp_kj_num_teamzk
            from {prefix}dm_zc_yw_moxing_team_info_m where {col_month_partition}={month}
            group by sig_offer_spec_id_teamzk
            ) b on a.sig_offer_spec_id_teamzk = b.sig_offer_spec_id_teamzk
            """)
    #  postgresql计算分位数: percentile_cont(0.25) within group (order by comp_wf_num_teamzk) spec_pre025_comp_wf_num_teamzk
    return sql


# </editor-fold-------------------------------------------------------------------------------------------------------->


#
#
# <editor-fold desc="移网目标字段表_套餐组 ---------------------------------------------------------------------------">
def ywteam_target_sql(month, prefix=prefix):
    month_now = sql_value(month)
    month_1next = sql_value(month_add(month, 1))
    month_2next = sql_value(month_add(month, 2))

    # truncate table {prefix}dm_zc_yw_moxing_team_target_m partition ({col_month_partition}={month});
    sql = sql_format(f"""
        insert overwrite table {prefix}dm_zc_yw_moxing_team_target_m partition ({col_month_partition}={month})
        select 
         a.{col_month_field}
        ,a.zk_user_no
        ,case when b.zk_user_no is null then 1 else 0 end flag_lw -- 离网

        ,case when b.zk_user_no is null then null when b.is_acct_teamzk='0' then 1 else 0 end flag_notacct -- 不出账

        ,case when b.zk_user_no is null then null when b.std_user_status_name_teamzk!='在用' then 1 else 0 end flag_badstatus -- 状态不正常

        ,case when b.zk_user_no is null or b.is_acct_teamzk='0' or b.std_user_status_name_teamzk!='在用' then 1 else 0 end flag_ls -- 移网流失

        ,case when b.zk_user_no is null or a.is_comp_teamzk='1' or substr(a.comp_offer_create_dt_teamzk,1,6)={month_now} then null 
              when b.is_comp_teamzk='1' or substr(b.comp_offer_create_dt_teamzk,1,6)>={month_1next} then 1 
              else 0 end flag_rh -- 升级融合

        ,case when b.zk_user_no is null or (a.is_comp_teamzk='0' and left(coalesce(a.comp_offer_create_dt_teamzk,'00000000'),6)!={month_now}) or a.comp_kj_num_teamzk>0 then null 
              when b.comp_kj_num_teamzk > a.comp_kj_num_teamzk then 1 
              else 0 end flag_kj -- 办理看家

        ,case when b.zk_user_no is null or (a.is_comp_teamzk='0'  and left(coalesce(a.comp_offer_create_dt_teamzk,'00000000'),6)!={month_now}) or a.comp_wf_num_teamzk>0 then null 
              when b.comp_wf_num_teamzk> a.comp_wf_num_teamzk then 1 
              else 0 end flag_wf -- 办理路由器

        ,case when b.fk_num_teamzk > a.fk_num_teamzk then 1 else 0 end flag_fk -- 加装副卡 

        ,case when not (a.is_acct_teamzk='1'and a.cert_type_teamzk='1'and a.is_red_list_teamzk='0'and a.open_months_teamzk>=3) or b.zk_user_no is null or c.zk_user_no is null then null 
              when ((left(b.sig_offer_create_dt_teamzk,6) ={month_1next} and a.sig_offer_spec_id_teamzk!=b.sig_offer_spec_id_teamzk) or (left(b.comp_offer_create_dt_teamzk,6) ={month_1next} and a.comp_offer_spec_id_teamzk!=b.comp_offer_spec_id_teamzk)) and c.arpu_teamadd>a.arpu_teamadd then 1  
              else 0 end flag_gq -- 套餐高迁 

        ,case when not (a.is_acct_teamzk='1'and a.cert_type_teamzk='1'and a.is_red_list_teamzk='0'and a.open_months_teamzk>=3) or b.zk_user_no is null or c.zk_user_no is null then null 
              when ((left(b.sig_offer_create_dt_teamzk,6) ={month_1next} and a.sig_offer_spec_id_teamzk!=b.sig_offer_spec_id_teamzk) or (left(b.comp_offer_create_dt_teamzk,6) ={month_1next} and a.comp_offer_spec_id_teamzk!=b.comp_offer_spec_id_teamzk)) and c.arpu_teamadd<a.arpu_teamadd then 1  
              else 0 end flag_down -- 套餐低迁

        ,case when not (a.is_acct_teamzk='1' and a.cert_type_teamzk='1' and a.is_red_list_teamzk='0' and a.open_months_teamzk>=3 and coalesce(a.pack_month_flow_num_teamadd,0)=0) then null 
              when b.pack_month_flow_num_m_teamadd>0 then 1 
              else 0 end flag_yb -- 流量月包

        ,case when not (a.is_acct_teamzk='1' and a.cert_type_teamzk='1' and a.is_red_list_teamzk='0' and a.open_months_teamzk>=3) then null 
              when b.is_new_tmn_teamzk='1' and b.start_use_time_teamzk>='{str(month) + '01'}' then 1 
              else 0 end flag_hj -- 终端换机

        ,case when not (a.is_acct_teamzk='1' and a.cert_type_teamzk='1' and a.is_red_list_teamzk='0' and a.open_months_teamzk>=3  and coalesce(a.pack_month_flow_num_teamadd,0)=0 and coalesce(a.pack_directed_num_teamadd,0)=0) then null 
              when b.pack_directed_num_teamadd>0 then 1 
              else 0 end flag_dxb -- 定向流量包

        ,case when not (a.is_acct_teamzk='1' and a.cert_type_teamzk='1' and a.is_red_list_teamzk='0' and a.open_months_teamzk>=3 and  a.sig_offer_spec_name_teamzk not like '%5g%' and a.pack_5g_internet_num_teamadd=0 and a.is_5g_main_offer_teamzk='0' and a.is_5g_upgrade_offer_teamzk='0') then null 
              when b.pack_5g_internet_num_teamadd>0 then 1 
              else 0 end flag_5gb --5g流量包

        ,case when not (a.is_acct_teamzk='1' and a.cert_type_teamzk='1' and a.is_red_list_teamzk='0' and a.open_months_teamzk>=3 and a.pack_spcl_num_teamadd=0) then null 
              when b.pack_spcl_num_teamadd>0 then 1 
              else 0 end flag_cl -- 彩铃

        ,null flag_1
        ,null flag_2
        ,null flag_3
        ,null flag_4  
        ,null flag_5  
        from (select * from {prefix}dm_zc_yw_moxing_team_info_m where {col_month_partition}={month_now}) as a
        left join (select * from {prefix}dm_zc_yw_moxing_team_info_m where {col_month_partition}={month_1next})as b on a.zk_user_no = b.zk_user_no
        left join (select * from {prefix}dm_zc_yw_moxing_team_info_m where {col_month_partition}={month_2next}) as c on c.zk_user_no = a.zk_user_no
        """)
    return sql


# </editor-fold-------------------------------------------------------------------------------------------------------->


#
#
# <editor-fold desc="宽带目标字段表 ----------------------------------------------------------------------------------">
def kduser_target_sql(month, prefix=prefix):
    month_now = sql_value(month)
    month_1next = sql_value(month_add(month, 1))

    # truncate table {prefix}dm_zc_kd_moxing_target_m partition ({col_month_partition}={month});
    sql = sql_format(f"""
        insert overwrite table {prefix}dm_zc_kd_moxing_target_m partition ({col_month_partition}={month})
        select 
        a.{col_month_field}
        ,a.user_no
        ,case when b.user_no is null then 1 else 0 end flag_lw -- 离网
        ,case when b.user_no is null then null when b.is_acct='0' then 1 else 0 end flag_notacct -- 不出账
        ,case when b.user_no is null then null when b.std_user_status_name!='在用' then 1 else 0 end flag_badstatus -- 状态不正常
        ,case when b.user_no is null or b.is_valid='0' or b.std_user_status_name!='在用' then 1 else 0 end flag_ls -- 宽带流失

        ,null flag_1
        ,null flag_2
        ,null flag_3
        ,null flag_4  
        ,null flag_5  
        from      (select * from {prefix}dm_zc_kd_moxing_info_m where {col_month_partition}={month_now}) a
        left join (select * from {prefix}dm_zc_kd_moxing_info_m where {col_month_partition}={month_1next}) b on a.user_no = b.user_no
        """)
    return sql


# </editor-fold-------------------------------------------------------------------------------------------------------->


#
#
# <editor-fold desc="融合表-------------------------------------------------------------------------------------------">
# 待填充
# </editor-fold-------------------------------------------------------------------------------------------------------->


# <editor-fold desc=" 统计数据库表中各个字段在某账期的取值分布 -------------------------------------------------------">
def get_tablefield(tablename):
    """
    根据basestr.py中的s_table_info，获取数据库表的字段名称及类型分类
    使用前提：python可连接数据库
    :param tablename: 数据库表名
    :return: DataFrame, 形如：
               field_name  dtype_classify
                 flag_ls         数值型
                 flag_rh         数值型
                 flag_fk         数值型
    """
    table_info = string_to_table(s_table_info)
    tablefield = DataFrame()
    for s_field_base in table_info.loc[table_info.tablename == tablename, 's_field_base'].unique():
        s_i = string_to_table(eval(s_field_base))
        s_i = s_i.loc[s_i.table == tablename, ['field_name', 'dtype_classify']]  # 限定某表的字段
        tablefield = pd.concat([tablefield, s_i])
    tablefield = tablefield.drop_duplicates()
    tablefield.index = range(len(tablefield))
    return tablefield


# tablename_in= tablename;tablefield= get_tablefield(tablename); dict_cutpoint=dict_cutpoints[tablename]; dict_classvalue=dict_classvalues[tablename]; sqlcomment=sqlcomment
def table_exam_sql(sqltype, month, tablename_in, tablefield, tablename_exam_v, dict_cutpoint=None, dict_classvalue=None,
                   col_month='acct_month', methods=['count_null', 'count_num0', 'count_numneg', 'num_Q_75'],
                   sqlcomment='----', union_count=50):
    """
    生成sql语句字符串：检查表中各个字段在各个账期的取值（可根据本地实际需求修改该函数）
    :param sqltype: sql的操作类型，execute：执行sql       print：打印sql（复制到数据库中执行）
    :param month:
    :param tablename_in: 待检查的数据库表名
    :param tablefield: DataFrame，tablename_in的数据字典（用到field_name、dtype_classify列）
                       形如：
                        field_name     dtype_classify
                        acct_month            类别型
                     phone_no_null            类别型
                               age            数值型
    :param tablename_exam_v: 检查结果表名（竖排）
    :param dict_cutpoint: 字典，key为字段名，value为分段切点
                          统计重点数值型字段的各分段的用户分布
                          dict_cutpoint取值为None则跳过此环节
    :param dict_classvalue: 字典，key为字段名，value为字段取值
                          统计重点类别型字段重点取值的用户分布
                          dict_classvalue为None则跳过此环节

    :param col_month: 账期字段名
    :param methods: 检查方式（根据实际需求调整、扩充）
        'count_null'：统计空置行数（全部字段）
        'count_num0'：统计零值行数（数值型字段）
        'count_numneg'：统计负值的行数（数值型字段）
        'num_Q_75'：统计75%分位数的取值（数值型字段）
    :param sqlcomment: sql的注释字符
    :param union_count: 按 union_count 分批插入数据
    :return: 在数据库中创建tablename_exam_v表
    """
    tablename_exam_v = tablename_exam_v.lower()

    if sqltype not in {'execute', 'print'}:
        raise Exception(f'sqltype参数取值为{sqltype}，应为execute、print')
    tablefield = tablefield.set_index('field_name').dtype_classify
    col_all = tablefield.index
    col_num = tablefield[tablefield == '数值型'].index

    partinfo = [(col_month, month, type_py_sql[str]), ('tablename', tablename_in, type_py_sql[str])]
    part, clear, insert = part_sql(tablename_exam_v, partinfo)

    if clear:
        print(f'{sqlcomment}创建/清空分区:{clear}')
        if sqltype == 'execute':
            my_sql_fun(clear, method='execute')

    overwrite = "first_no_replace"

    if dict_cutpoint:
        cut_more = set(dict_cutpoint.keys()) - set(tablefield.index)
        if cut_more:
            raise Exception(f"dict_cutpoint的下列key 不在 数据库表的字段列表中：{cut_more}")

        for col, cutpoint in dict_cutpoint.items():  # col, cutpoint = list(dict_cutpoint.items())[0]
            if cutpoint[0] == float("-inf"):
                case = f'case when {col} <= {cutpoint[1]}'
            else:
                case = f'case when {col} >= {cutpoint[0]} and {col} <= {cutpoint[1]}'
            case += f" then 'count_[{cutpoint[0]}, {cutpoint[1]}]'"

            for idx in range(2, len(cutpoint) - 1):
                case += f"\nwhen {col} <= {cutpoint[idx]} then 'count_({cutpoint[idx - 1]}, {cutpoint[idx]}]'"

            if len(cutpoint) > 2:
                if cutpoint[-1] == float("inf"):
                    case += f'\nwhen {col} > {cutpoint[-2]}'
                else:
                    case += f'\nwhen {col} > {cutpoint[-2]} and col<={cutpoint[-1]}'
                case += f" then 'count_({cutpoint[-2]}, {cutpoint[-1]}]'"
            case += "else 'count_其他区间' \nend col_stat"

            count_cp_sql = sql_format(f"""
            {insert.replace(overwrite, "into")}
            select '{month}' {col_month}, '{tablename_in}' tablename, '{col}' col_name, col_stat, count(1) col_value from
            (
            select {case} from {tablename_in}
            where {col_month}={sql_value(month)}
            ) t group by col_stat
            """)
            sql_show(f"\n\n\n{sqlcomment} {col}分段统计sql {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:",
                     count_cp_sql)
            if sqltype == 'execute':
                my_sql_fun(count_cp_sql, method='execute')
            overwrite = "overwrite table"

    if dict_classvalue:
        cut_more = set(dict_cutpoint.keys()) - set(tablefield.index)
        if cut_more:
            raise Exception(f"dict_classvalue的下列key 不在 数据库表的字段列表中：{cut_more}")

        for col, classvalue in dict_classvalue.items():  # col, classvalue = list(dict_classvalue.items())[0]
            count_class_sql = sql_format(f"""
            {insert.replace(overwrite, "into")}
            select '{month}' {col_month}, '{tablename_in}' tablename, '{col}' col_name, concat('count_', {col}), count(1) col_stat 
            from {tablename_in} 
            where {col_month}={sql_value(month)} and {col} in ({', '.join([sql_value(i) for i in classvalue])}) 
            group by {col}
            """)
            sql_show(
                f"\n\n\n{sqlcomment} {col}重点取值统计sql {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:",
                count_class_sql)
            if sqltype == 'execute':
                my_sql_fun(count_class_sql, method='execute')
            overwrite = "overwrite table"

    def cut_union_sql(null_sqls):
        count_sql, j = ['', 0]
        for i in [*range(0, len(null_sqls), union_count), len(null_sqls)][1:]:
            # print(i)
            _list = null_sqls[j:i]
            _union_null_sqls = '\nunion all\n'.join(_list)
            _count_null_sql = sql_format(f"""
                                {insert.replace(overwrite, "into")}
                                {_union_null_sqls}
                                """)
            j = i
            count_sql += '\n\n' + _count_null_sql + ';'
        return count_sql

    if 'count_null' in methods:
        null_sqls = [sql_format(f"""
            select '{month}' {col_month}, '{tablename_in}' tablename, '全量' col_name, 'count_全量' col_stat, count(1) col_value
            from {tablename_in} where {col_month}={sql_value(month)}""")]

        for i in col_all:
            null_sqls_i = sql_format(f"""
                select '{month}' {col_month}, '{tablename_in}' tablename, '{i}' col_name, 'count_空值' col_stat, sum(case when {i} is null then 1 else 0 end) col_value
                from {tablename_in} where {col_month}={sql_value(month)}""")
            null_sqls.append(null_sqls_i)

        count_null_sql = cut_union_sql(null_sqls)

        sql_show(f"\n\n\n{sqlcomment}空值统计sql {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:",
                 count_null_sql)
        if sqltype == 'execute':
            my_sql_fun(count_null_sql, method='execute')
        overwrite = "overwrite table"

    if 'count_num0' in methods:
        num0_sqls = []
        for i in col_num:
            num0_sqls_i = sql_format(f"""
                select '{month}' {col_month}, '{tablename_in}' tablename, '{i}' col_name, 'count_零值' col_stat, sum(case when {i}=0 then 1 else 0 end) col_value
                from {tablename_in} where {col_month}={sql_value(month)}""")
            num0_sqls.append(num0_sqls_i)

        count_num0_sqls = cut_union_sql(num0_sqls)

        sql_show(f"\n\n\n{sqlcomment}零值(数值字段)统计sql {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:",
                 count_num0_sqls)
        if sqltype == 'execute':
            my_sql_fun(count_num0_sqls, method='execute')
        overwrite = "overwrite table"

    if 'count_numneg' in methods:
        numneg_sqls = []
        for i in col_num:
            numneg_sqls_i = sql_format(f"""
                select '{month}' {col_month}, '{tablename_in}' tablename, '{i}' col_name, 'count_负值' col_stat, sum(case when {i}<0 then 1 else 0 end) col_value
                from {tablename_in} where {col_month}={sql_value(month)}""")
            numneg_sqls.append(numneg_sqls_i)
        count_numneg_sqls = cut_union_sql(numneg_sqls)

        sql_show(f"\n\n\n{sqlcomment}负值(数值字段)统计sql {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:",
                 count_numneg_sqls)
        if sqltype == 'execute':
            my_sql_fun(count_numneg_sqls, method='execute')
        overwrite = "overwrite table"

    Qs = [i for i in methods if 'num_Q_' in i]
    for q in Qs:
        Q_p = int(q.replace('num_Q_', ''))
        p = Q_p / 100
        Q_sqls = []
        for i in col_num:
            Q_sqls_i = sql_format(f"""
                select '{month}' {col_month}, '{tablename_in}' tablename, '{i}' col_name, '分位数_{Q_p}%' col_stat, {percentile_fun(i, p)} col_value
                from {tablename_in} where {col_month}={sql_value(month)}""")
            Q_sqls.append(Q_sqls_i)
        count_Q_sqls = cut_union_sql(Q_sqls)

        sql_show(
            f"\n\n\n{sqlcomment}{Q_p}%分位数(数值字段)统计sql {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:",
            count_Q_sqls)
        if sqltype == 'execute':
            my_sql_fun(count_Q_sqls, method='execute')
        overwrite = "overwrite table"

    method_more = set([i for i in methods if 'num_Q_' not in i]) - {'count_null', 'count_num0', 'count_numneg'}
    if method_more:
        raise Exception(f'methods中{method_more}未处理，请扩充代码！')


def table_exam_vtoh(sqltype, month_list, tablename_exam_v, tablename_exam_h, col_month='acct_month', nmg_yaxin=('', ''),
                    bigdiff_sql="exam_sep>0.1", sqlcomment='----'):
    """
    将竖排的检查表转为横表，方便对比查看，并判断数据分布是否有异常
    :param sqltype: sql的操作类型，execute：执行sql       print：打印sql（复制到数据库中执行）
    :param month_list: 检查的账期列表
    :param tablename_exam_v: 检查结果表名（竖排）
    :param tablename_exam_h: 检查结果表名（横排）
    :param col_month: 账期字段名
    :param bigdiff_sql: “分布差异大”的判断条件
    :param sqlcomment: sql的注释字符
    :return: 在数据库中创建tablename_exam_h表
    """
    tablename_exam_v = tablename_exam_v.lower()
    tablename_exam_h = tablename_exam_h.lower()
    month_list_comma = ', '.join([sql_value(i) for i in month_list])

    sql_rowcol = sql_format(f"""
        select tablename, acct_month, 
        sum(case when col_name='全量' then col_value else 0 end) count_row,
        count(1) count_columns
        from {tablename_exam_v}
        where {col_month} in ({month_list_comma})
        group by tablename, acct_month
        order by tablename, acct_month""")
    sql_show(f'\n\n{sqlcomment}各表各账期行列数统计', sql_rowcol)
    if sqltype == 'execute':
        row_col = my_sql_fun(sql_rowcol, method='read')
        print(f"\n查询结果：\n{row_col}")

    sql_stat = sql_format(f"""
        select col_stat,
        count(1)
        from {tablename_exam_v}
        group by col_stat 
        order by count(1) desc""")
    sql_show(f"\n\n{sqlcomment}方式统计种类", sql_stat)
    if sqltype == 'execute':
        stat = my_sql_fun(sql_stat, method='read')
        print(f"\n查询结果：\n{stat}")

    t1_col_values = re.sub('^ *', '', ', \n'.join([f"{' ' * 8}t1.col_value_{i}" for i in month_list]))
    # 用户量占比为100%：1， 0%：0， 0%与100%：x
    sql_exam_ratio = re.sub('^ *', '', ',\n'.join(
        [f"{' ' * 25}case t1.col_value_{i}/c.count_all_{i} when 1 then '1' when 0 then '0' else 'x' end" for i in
         month_list]))
    col_value_toh = re.sub('^ *', '', ',\n'.join(
        [f"{' ' * 12}sum(case when {col_month}={sql_value(i)} then col_value else 0 end) col_value_{i}" for i in
         month_list]))
    sql_count_all = re.sub('^ *', '', ',\n'.join([
                                                     f"{' ' * 12}sum(case when {col_month}={sql_value(i)} and col_name='全量' then col_value else 0 end) count_all_{i}"
                                                     for i in month_list]))
    sql_vtoh = sql_format(f"""
    drop table if exists {tablename_exam_h};
    create {nmg_yaxin[0]} table {tablename_exam_h} {nmg_yaxin[1].replace('%s', tablename_exam_h)} as
    select *,	 
    case when exam_ratio='1111' then (case col_stat when 'count_负值' then '' when'count_空值' then '所有账期全空'  when 'count_零值' then '部分账期全零' else '待扩充1' end)
         when exam_ratio not in ('', '0000', 'xxxx', '1111') then '分布不同'
         when {bigdiff_sql} then '分布差异大'
         else ''
         end exam_result
    from (
        select t1.tablename, t1.col_name, t1.col_stat,
        {t1_col_values},
        case when t1.col_stat not like 'count_%' then ''
             when t1.col_name='全量' then ''
             else concat({sql_exam_ratio}
                        )
             end exam_ratio,
        s.exam_std,
        round(case when s.exam_avg=0 then 0 else s.exam_std/s.exam_avg end, 3) exam_sep
        from (
            select tablename, col_name, col_stat,
            {col_value_toh}
            from {tablename_exam_v} 
            where {col_month} in ({month_list_comma})
            group by tablename, col_name, col_stat
            ) t1 
        left join (
            select tablename,
            {sql_count_all}
            from {tablename_exam_v}
            where {col_month} in ({month_list_comma})
            group by tablename
            ) c on t1.tablename=c.tablename
        left join (
            select tablename, col_name, col_stat,
            round(stddev(col_value), 3) exam_std,
            avg(col_value) exam_avg
            from {tablename_exam_v}
            where {col_month} in ({month_list_comma})
            group by tablename, col_name, col_stat
            ) s on t1.tablename=s.tablename and t1.col_name=s.col_name and t1.col_stat=s.col_stat
    ) t	  """)
    sql_show(f'\n\n{sqlcomment}竖表转横表方便查看，并检查数据分布', sql_vtoh)
    if sqltype == 'execute':
        my_sql_fun(sql_vtoh, method='execute')

    sql_result = f"select * from {tablename_exam_h} order by exam_result desc, exam_sep desc"
    sql_show(f"\n\n{sqlcomment}查看检查结果(根据实际需求修改此句)\n", sql_result)
    if sqltype == 'execute':
        res = my_sql_fun(sql_result, method='read')
        print(f"查看前几行：\n{res.head()}")


# </editor-fold-------------------------------------------------------------------------------------------------------->


#
#
# <editor-fold desc="创建分区表: 移网宽表_套餐组 ---------------------------------------------------------------------">
tablename_cre_team_summary = "kehujingyingbudb.dm_zc_yw_moxing_team_info_m"
cre_team_summary = """
drop table if exists @tablename@;
CREATE external TABLE @tablename@(
  month_id VARCHAR(6),
  zk_user_no VARCHAR(20),
  user_no_teamzk STRING,
  acct_id_teamzk STRING,
  customer_no_teamzk STRING,
  dev_channel_nbr_teamzk STRING,
  dev_channel_name_teamzk STRING,
  open_date_teamzk STRING,
  open_months_teamzk INT,
  cert_type_teamzk STRING,
  cert_nbr_teamzk STRING,
  area_no_teamzk STRING,
  city_no_teamzk STRING,
  is_village_teamzk STRING,
  age_teamzk STRING,
  sex_teamzk STRING,
  is_valid_teamzk STRING,
  std_user_status_name_teamzk STRING,
  stop_times_teamadd BIGINT,
  resent_stop_date_teammax STRING,
  is_acct_teamzk STRING,
  payment_mode_cd_teamzk STRING,
  is_real_name_teamzk STRING,
  is_gz_teamzk STRING,
  is_xy_teamzk STRING,
  is_gov_teamzk STRING,
  is_grp_member_teamzk STRING,
  is_hlwk_teamzk STRING,
  is_sw_teamzk STRING,
  is_red_list_teamzk STRING,
  is_5g_main_offer_teamzk STRING,
  is_5g_upgrade_offer_teamzk STRING,
  is_fq_teamzk STRING,
  is_usim_teamzk STRING,
  tyaddresscode_teamzk STRING,
  tyaddressname_teamzk STRING,
  shentoulv_teamzk STRING,
  family_form_ersj_teamzk STRING,
  family_form_skzj_teamzk STRING,
  family_form_unknown_teamzk STRING,
  family_prop_child_teamzk STRING,
  family_prop_elder_teamzk STRING,
  family_prop_unknown_teamzk STRING,
  family_type_teamzk STRING,
  sig_offer_spec_id_teamzk STRING,
  sig_offer_spec_name_teamzk STRING,
  sig_offer_create_dt_teamzk STRING,
  comp_offer_spec_id_teamzk STRING,
  comp_offer_spec_name_teamzk STRING,
  comp_offer_create_dt_teamzk STRING,
  is_dupety_teamzk STRING,
  fk_num_teamzk INT,
  prod_spec_fee_teamzk DECIMAL(20,4),
  prod_spec_fluw_teamzk DECIMAL(20,4),
  prod_spec_dur_teamzk DECIMAL(20,4),
  mbl_inner_fluw_teamadd DECIMAL(30,4),
  dialing_inner_teamadd DECIMAL(30,4),
  is_comp_teamzk STRING,
  comp_user_no_teamzk STRING,
  comp_eff_date_teamzk STRING,
  comp_exp_date_teamzk STRING,
  comp_type_teamzk STRING,
  comp_num_teamzk INT,
  comp_kd_num_teamzk INT,
  comp_yd_num_teamzk INT,
  comp_hd_num_teamzk INT,
  comp_wf_num_teamzk INT,
  comp_kj_num_teamzk INT,
  is_agre_teamzk STRING,
  track_eff_date_teamzk STRING,
  track_exp_date_teamzk STRING,
  pack_hd_num_teamadd BIGINT,
  pack_dur_num_teamadd BIGINT,
  pack_flow_num_teamadd BIGINT,
  pack_flow_sum_teamadd DECIMAL(30,4),
  pack_dur_sum_teamadd DECIMAL(30,4),
  pack_month_flow_num_teamadd BIGINT,
  pack_month_flow_sum_teamadd DECIMAL(30,4),
  pack_month_flow_num_m_teamadd BIGINT,
  pack_month_flow_sum_m_teamadd DECIMAL(30,4),
  pack_month_flow_exp_date_teamzk STRING,
  pack_directed_num_teamadd BIGINT,
  pack_directed_sum_teamadd DECIMAL(30,4),
  pack_directed_num_m_teamadd BIGINT,
  pack_directed_sum_m_teamadd DECIMAL(30,4),
  pack_directed_exp_date_teamzk STRING,
  pack_5g_internet_num_teamadd BIGINT,
  pack_5g_internet_free_num_teamadd BIGINT,
  pack_5g_internet_sum_teamadd DECIMAL(30,4),
  pack_5g_internet_num_m_teamadd BIGINT,
  pack_5g_internet_free_num_m_teamadd BIGINT,
  pack_5g_internet_sum_m_teamadd DECIMAL(30,4),
  pack_5g_internet_exp_date_teamzk STRING,
  pack_spcl_num_teamadd BIGINT,
  pack_spcl_free_num_teamadd BIGINT,
  pack_spcl_num_m_teamadd BIGINT,
  pack_spcl_free_num_m_teamadd BIGINT,
  pack_spcl_exp_date_teamzk STRING,
  pack_spcl_6_month_teammax STRING,
  is_hy_or_hb_teamzk STRING,
  point_teamzk DECIMAL(20,4),
  star_teamzk DECIMAL(20,4),
  basic_credit_teamzk DECIMAL(20,4),
  quota_teamzk STRING,
  is_lh_teamzk STRING,
  lh_type_teamzk STRING,
  is_bxl_teamzk STRING,
  bxl_flow_step_teamzk STRING,
  bxl_deal_times_teamzk INT,
  is_deal_bxl_teamzk STRING,
  pack_bxl_num_teamzk INT,
  owe_flag_teamzk STRING,
  owe_charge_teamadd DECIMAL(30,4),
  owe_times_teamadd BIGINT,
  last_owe_month_teammax STRING,
  last_owe_charge_teamzk DECIMAL(20,4),
  owe_charge_acct_teamadd DECIMAL(30,4),
  payment_cnt_teamadd DECIMAL(30,4),
  payment_charge_teamadd DECIMAL(30,4),
  payment_time_teammax STRING,
  payment_fee_teamzk DECIMAL(20,4),
  balance_teamadd DECIMAL(30,4),
  arpu_teamadd DECIMAL(30,4),
  call_fee_teamadd DECIMAL(30,4),
  call_local_fee_teamadd DECIMAL(30,4),
  call_long_prov_fee_teamadd DECIMAL(30,4),
  call_roam_fee_teamadd DECIMAL(30,4),
  flux_fee_teamadd DECIMAL(30,4),
  sms_fee_teamadd DECIMAL(30,4),
  rgst_tmn_no_teamzk STRING,
  rgst_tmn_brand_teamzk STRING,
  rgst_tmn_model_teamzk STRING,
  rgst_tmn_type_teamzk STRING,
  rgst_tmn_time_teamzk STRING,
  rgst_tmn_flag_teamzk STRING,
  phone_price_teamzk STRING,
  operating_sys_teamzk STRING,
  market_time_teamzk STRING,
  prd_position_teamzk STRING,
  screens_nbr_teamzk STRING,
  main_screen_size_teamzk STRING,
  baseband_chip_clocked_teamzk STRING,
  resolution_teamzk STRING,
  ram_teamzk STRING,
  main_camera_teamzk STRING,
  start_use_time_teamzk STRING,
  end_use_time_teamzk STRING,
  usage_days_teamzk INT,
  usage_days_sum_teamzk INT,
  is_in_use_teamzk STRING,
  is_new_tmn_teamzk STRING,
  rgst_tmn_brand_1_teamzk STRING,
  rgst_tmn_model_1_teamzk STRING,
  rgst_tmn_type_1_teamzk STRING,
  rgst_tmn_time_1_teamzk STRING,
  rgst_tmn_flag_1_teamzk STRING,
  phone_price_1_teamzk STRING,
  operating_sys_1_teamzk STRING,
  market_time_1_teamzk STRING,
  prd_position_1_teamzk STRING,
  screens_nbr_1_teamzk STRING,
  main_screen_size_1_teamzk STRING,
  baseband_chip_clocked_1_teamzk STRING,
  resolution_1_teamzk STRING,
  ram_1_teamzk STRING,
  main_camera_1_teamzk STRING,
  start_use_time_1_teamzk STRING,
  end_use_time_1_teamzk STRING,
  usage_days_1_teamzk INT,
  usage_days_sum_1_teamzk INT,
  is_new_tmn_1_teamzk STRING,
  rgst_tmn_brand_2_teamzk STRING,
  rgst_tmn_model_2_teamzk STRING,
  rgst_tmn_type_2_teamzk STRING,
  rgst_tmn_time_2_teamzk STRING,
  rgst_tmn_flag_2_teamzk STRING,
  phone_price_2_teamzk STRING,
  operating_sys_2_teamzk STRING,
  market_time_2_teamzk STRING,
  prd_position_2_teamzk STRING,
  screens_nbr_2_teamzk STRING,
  main_screen_size_2_teamzk STRING,
  baseband_chip_clocked_2_teamzk STRING,
  resolution_2_teamzk STRING,
  ram_2_teamzk STRING,
  main_camera_2_teamzk STRING,
  start_use_time_2_teamzk STRING,
  end_use_time_2_teamzk STRING,
  usage_days_2_teamzk INT,
  usage_days_sum_2_teamzk INT,
  is_new_tmn_2_teamzk STRING,
  is_shaungka_teamzk STRING,
  shaungka_type_teamzk STRING,
  dur_teamadd DECIMAL(30,4),
  calling_dur_teamadd DECIMAL(30,4),
  called_dur_teamadd DECIMAL(30,4),
  cnt_teamadd BIGINT,
  calling_cnt_teamadd BIGINT,
  called_cnt_teamadd BIGINT,
  roam_dur_teamadd DECIMAL(30,4),
  roam_cnt_teamadd DECIMAL(30,4),
  roam_out_dur_teamadd DECIMAL(30,4),
  roam_dmstc_dur_teamadd DECIMAL(30,4),
  local_dur_teamadd DECIMAL(30,4),
  call_days_teammax INT,
  calling_days_teammax INT,
  called_days_teammax INT,
  roam_out_days_teammax INT,
  roam_dmstc_days_teammax INT,
  local_days_teammax INT,
  ct_dur_teamadd DECIMAL(30,4),
  wj_dur_teamadd DECIMAL(30,4),
  ct_calling_dur_teamadd DECIMAL(30,4),
  ct_cnt_teamadd DECIMAL(30,4),
  ct_calling_cnt_teamadd DECIMAL(30,4),
  wj_calling_dur_teamadd DECIMAL(30,4),
  wj_cnt_teamadd DECIMAL(30,4),
  wj_calling_cnt_teamadd DECIMAL(30,4),
  jwq_num_teamadd BIGINT,
  jwq_ct_num_teamadd BIGINT,
  jwq_wj_num_teamadd BIGINT,
  jwq_calling_num_teamadd BIGINT,
  jwq_ct_calling_num_teamadd BIGINT,
  jwq_wj_calling_num_teamadd BIGINT,
  jwq_called_num_teamadd BIGINT,
  jwq_ct_called_num_teamadd BIGINT,
  jwq_wj_called_num_teamadd BIGINT,
  data_flux_teamadd DECIMAL(30,4),
  last_to_acct_flux_teamzk DECIMAL(20,4),
  acct_to_next_flux_teamadd DECIMAL(30,4),
  flux_5g_teamadd DECIMAL(30,4),
  flux_4g_teamadd DECIMAL(30,4),
  flux_3g_teamadd DECIMAL(30,4),
  flux_2g_teamadd DECIMAL(30,4),
  int_flux_teamadd DECIMAL(30,4),
  dmstc_flux_teamadd DECIMAL(30,4),
  flux_days_teammax INT,
  int_flux_days_teammax INT,
  dmstc_flux_days_teammax INT,
  normal_flux_teamadd DECIMAL(30,4),
  normal_flux_days_teammax INT,
  night_flux_teamadd DECIMAL(30,4),
  night_flux_days_teammax INT,
  weekday_flux_avg_teamadd DECIMAL(30,4),
  weekend_flux_avg_teamadd DECIMAL(30,4),
  sx_flux_avg_teamadd DECIMAL(30,4),
  zx_flux_avg_teamadd DECIMAL(30,4),
  xx_flux_avg_teamadd DECIMAL(30,4),
  mbl_out_flow_teamadd DECIMAL(30,4),
  sms_cnt_teamadd BIGINT,
  sms_send_cnt_teamadd BIGINT,
  sms_recv_cnt_teamadd BIGINT,
  self_scs_cnt_teamadd BIGINT,
  self_scs_tsbz_cnt_teamadd BIGINT,
  call_turn_cnt_teamadd BIGINT)
stored as orc
location 'hdfs://hacluster/warehouse/tablespace/external/hive/kehujingyingbudb.db/@tablename@'
partitioned by (acct_month string)""".replace('@tablename@', tablename_cre_team_summary)

# print(cre_team_summary)
# my_sql_fun(cre_team_summary, method='execute')
# </editor-fold-------------------------------------------------------------------------------------------------------->


#
#
# <editor-fold desc="创建分区表: 移网补充特征表_套餐组 ---------------------------------------------------------------">
tablename_cre_ywteam_xadd = "kehujingyingbudb.dm_zc_yw_moxing_team_add_m"
cre_ywteam_xadd = """
drop table if exists @tablename@;
CREATE external TABLE @tablename@ (
  month_id VARCHAR(6),
  zk_user_no VARCHAR(20),
  sig_offer_spec_id_teamzk_count BIGINT,
  spec_count0_fk_num_teamzk BIGINT,
  spec_count1_fk_num_teamzk BIGINT,
  spec_count2_fk_num_teamzk BIGINT,
  spec_pre025_fk_num_teamzk INT,
  spec_pre05_fk_num_teamzk INT,
  spec_max_fk_num_teamzk INT,
  spec_count0_comp_wf_num_teamzk BIGINT,
  spec_count1_comp_wf_num_teamzk BIGINT,
  spec_count2_comp_wf_num_teamzk BIGINT,
  spec_pre025_comp_wf_num_teamzk INT,
  spec_pre05_comp_wf_num_teamzk INT,
  spec_max_comp_wf_num_teamzk INT,
  spec_count0_comp_kj_num_teamzk BIGINT,
  spec_count1_comp_kj_num_teamzk BIGINT,
  spec_count2_comp_kj_num_teamzk BIGINT,
  spec_pre025_comp_kj_num_teamzk INT,
  spec_pre05_comp_kj_num_teamzk INT,
  spec_max_comp_kj_num_teamzk INT)
stored as orc
location 'hdfs://hacluster/warehouse/tablespace/external/hive/kehujingyingbudb.db/@tablename@'
partitioned by (acct_month string)""".replace('@tablename@', tablename_cre_ywteam_xadd)

# print(cre_ywteam_xadd)
# my_sql_fun(cre_ywteam_xadd, method='execute')
# </editor-fold-------------------------------------------------------------------------------------------------------->


#
#
# <editor-fold desc="创建分区表: 移网目标字段表_套餐组 ---------------------------------------------------------------">
tablename_cre_ywteam_target = "kehujingyingbudb.dm_zc_yw_moxing_team_target_m"
cre_ywteam_target = """
drop table if exists @tablename@;
CREATE external TABLE @tablename@(
  month_id VARCHAR(6),
  zk_user_no VARCHAR(20),
  flag_lw INT,
  flag_notacct INT,
  flag_badstatus INT,
  flag_ls INT,
  flag_rh INT,
  flag_kj INT,
  flag_wf INT,
  flag_fk INT,
  flag_gq INT,
  flag_down INT,
  flag_yb INT,
  flag_hj INT,
  flag_dxb INT,
  flag_5gb INT,
  flag_cl INT,
  flag_1 INT,
  flag_2 INT,
  flag_3 INT,
  flag_4 INT,
  flag_5 INT)
stored as orc
location 'hdfs://hacluster/warehouse/tablespace/external/hive/kehujingyingbudb.db/@tablename@'
partitioned by (acct_month string)""".replace('@tablename@', tablename_cre_ywteam_target)

# print(cre_ywteam_target)
# my_sql_fun(cre_ywteam_target, method='execute')
# </editor-fold-------------------------------------------------------------------------------------------------------->


#
#
# <editor-fold desc="创建分区表: 宽带目标字段表_套餐组 ---------------------------------------------------------------">
tablename_cre_kdteam_target = "kehujingyingbudb.dm_zc_kd_moxing_target_m"
cre_kdteam_target = """
drop table if exists @tablename@;
CREATE external TABLE @tablename@(
  month_id VARCHAR(6),
  user_no VARCHAR(20),
  flag_lw INT,
  flag_notacct INT,
  flag_badstatus INT,
  flag_ls INT,

  flag_1 INT,
  flag_2 INT,
  flag_3 INT,
  flag_4 INT,
  flag_5 INT)
stored as orc
location 'hdfs://hacluster/warehouse/tablespace/external/hive/kehujingyingbudb.db/@tablename@'
partitioned by (acct_month string)""".replace('@tablename@', tablename_cre_kdteam_target)

# print(cre_kdteam_target)
# my_sql_fun(cre_kdteam_target, method='execute')
# </editor-fold-------------------------------------------------------------------------------------------------------->


#
#
# <editor-fold desc="创建分区表: 数据质量检查表(tablename_exam_v 竖表) -----------------------------------------------">
if db == 'gp':
    cre_exam_v = """
    drop table if exists @tablename@;
    CREATE TABLE @tablename@(
      acct_month text,
      tablename text,
      col_name text,
      col_stat text,
      col_value numeric)
    partition by list(acct_month)"""
elif db == 'hive':
    # hive中账期和模型名称与分区键重复,防止误操作插错数据而不得知
    cre_exam_v = """
    drop table if exists @tablename@;
    CREATE external TABLE @tablename@(
      month_id string,
      table_name string,
      col_name string,
      col_stat string,
      col_value DECIMAL(30,4))
    stored as orc
    location 'hdfs://hacluster/warehouse/tablespace/external/hive/kehujingyingbudb.db/@tablename@'
    partitioned by (acct_month string, tablename string)"""

# </editor-fold-------------------------------------------------------------------------------------------------------->
