#<editor-fold desc="本模块说明">
# 由于很多项目并不支持将文件（excel等）上传至建模环境，故从excel中复制、粘贴得到长字符串，
# 后续将由string_to_table（Mining.selfmodule.toolmodule.strtotable）转换为DataFrame
# </editor-fold>

from Mining.selfmodule.toolmodule.datatrans import sql_format
from pandas import DataFrame


# ------------------------------ <editor-fold desc="模型信息_静态"> ---------------------------------------------------
# condition_base_dict：移网、宽带、融合等用户群基本限制条件：转换为模型信息中的condition_base列，用于./selfmodule/binarymodule/modelinfo.py中的privy_modelsummary()
# 1 移网类所有模型、宽带类所有模型、融合类所有模型共同的用户基本限制条件，按照sql语法编写（字段名小写：有些项目建模环境下可能会在python内存中对读取进来的DataFrame进行事后筛选）
# 2 如果使用日数据限制用户范围：在日数据字段前添加dayvalue_前缀，以与月数据字段区分，因为二者可能存在同名字段
#   2.1 限制观察期次月dd日 在网、状态正常等
#   2.2 若在日数据到位前打分，日数据条件失效（dayvalue_），在清单输出时补充！！！
#   2.3 流失模型：如果打分时不想按观察期月底限制状态正常用户，只想限制清单输出日时状态正常的，
#                 可在condition_base中删除月数据的状态正常条件，只在condition_add中添加日数据的状态正常限制，
#                 如此，可输出观察期月底状态不正常，但清单输出时状态正常的用户，这部分用户的流失率极有可能更高。
#   2.4 其他模型：condition_add中添加月、日数据两个状态正常条件（状态一直正常的用户办理活动的可能性更高）。
#   2.5 在网条件：在condition_base中限制月、日数据在网条件，这样即使打分时日数据条件失效，月底已离网用户也不会进入打分范围
condition_base_dict = dict()

# 模型示例的基础条件
condition_base_dict['con_fake'] = """
    phone_no_null is null and
    last_stop_date is not null and
    innet_months >= 3 and
    dayvalue_gprs_flow>0
    """
# condition_base_dict['con_fake'] = """
#     phone_no_null is null and
#     dayvalue_phone_no_null is null and
#     last_stop_date is not null and
#     innet_months >= 3"""

# 所有基于移网用户的模型共同的用户限制条件（sql）
condition_base_dict['con_yw_team'] = sql_format("""
    cert_type_teamzk='1' and 
    is_valid_teamzk='1' and 
    is_acct_teamzk='1' and 
    is_red_list_teamzk='0' and
    dayvalue_std_user_status='100000' and 
    open_months_teamzk >= 3""")

# 所有基于宽带用户的模型共同的用户限制条件（sql）
condition_base_dict['con_kd_user'] = sql_format("""
    is_valid='1' and 
    is_red_list='0' and 
    dayvalue_std_user_status='100000' and 
    open_months >= 3
    """).replace('\n', '')  # cert_type='1'
# postgresql: 1+((month_id::VARCHAR||'01')::date - open_date::date)/30.0 >= 3

# 所有基于融合用户的模型共同的用户限制条件（sql）
condition_base_dict['con_rh_team'] = condition_base_dict['con_yw_team']

# 从excel粘贴过来的模型信息_静态，用于./selfmodule/binarymodule/modelinfo.py中的privy_modelsummary()
s_info_quiet = """model_name	short_name	table_XY	s_field_base	condition_base	condition_add	available_add	col_month	col_id	col_target	Pcase	Ncase	target_lag	col_stratified	col_out	dict_sortscore	score_targetnull
模型示例     	eg	kehujingyingbudb.ml_xy_eg_m	s_yw_fake	con_fake	user_status='在网-正常'	available_notzd	acct_month	user_id	flag_eg	1	0	1	np.nan	['acct_month']	{'sms_cnt': True, 'call_fee_local': False}	Ncase
模型示例2     	eg2	kehujingyingbudb.ml_xy_eg_m	s_yw_fake	con_fake	user_status='在网-正常'	available_notzd	acct_month	user_id	flag_eg2	'1'	'0'	1	np.nan	['acct_month']	{'sms_cnt': True, 'call_fee_local': False}	Ncase
流失预警模型_移网	lsyw	kehujingyingbudb.ml_xy_yw_m_new	s_yw_team	con_yw_team	is_agre_teamzk='0'	available_notzd	acct_month	zk_user_no	flag_ls	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': True, 'data_flux_teamadd': True}	Pcase
流失预警模型_宽带	lskd	kehujingyingbudb.ml_xy_kd_m	s_kd_user	con_kd_user	np.nan	np.nan	acct_month	user_no	flag_ls	1	0	1	np.nan	['acct_month', 'area_no']	{'arpu': True, 'data_dur': True}	Pcase
转融合模型_单C	rhdc	kehujingyingbudb.ml_xy_yw_m_new	s_yw_team	con_yw_team	dayvalue_is_agre='0' and dayvalue_is_comp='0' and (substr(comp_offer_create_dt_teamzk,1,6)!=acct_month or comp_offer_create_dt_teamzk is null)	available_notzd	acct_month	zk_user_no	flag_rh	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
副卡加装模型_移网	fkyw	kehujingyingbudb.ml_xy_yw_m_new	s_yw_team	con_yw_team	 fk_num_teamzk in (0, 1) and is_hlwk_teamzk='0' and prod_spec_fee_teamzk>=19 and substr(sig_offer_create_dt_teamzk,1,4)>'2015'	available_notzd	acct_month	zk_user_no	flag_fk	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
看家潜客模型_融合	kjrh	kehujingyingbudb.ml_xy_rh_m	s_rh_team	con_rh_team	dayvalue_is_comp='1' and comp_kj_num_teamzk=0	np.nan	acct_month	zk_user_no	flag_kj	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
路由器潜客模型_融合	wfrh	kehujingyingbudb.ml_xy_rh_m	s_rh_team	con_rh_team	dayvalue_is_comp='1' and comp_wf_num_teamzk=0	np.nan	acct_month	zk_user_no	flag_wf	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
套餐高迁模型	gqyw	kehujingyingbudb.ml_xy_yw_m_new	s_yw_team	con_yw_team	 is_fq_teamzk!='1' 	available_notzd	acct_month	zk_user_no	flag_gq	1	0	2	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
流量月包加装模型	ybyw	kehujingyingbudb.ml_xy_yw_m_new	s_yw_team	con_yw_team	 pack_month_flow_num_teamadd=0	available_notzd	acct_month	zk_user_no	flag_yb	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
终端换机模型	hjyw	kehujingyingbudb.ml_xy_yw_m_new	s_yw_team	con_yw_team	 np.nan	available_zd	acct_month	zk_user_no	flag_hj	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
加定向流量包模型	dxbyw	kehujingyingbudb.ml_xy_yw_m_new	s_yw_team	con_yw_team	 pack_month_flow_num_teamadd=0 and pack_directed_num_teamadd=0	available_notzd	acct_month	zk_user_no	flag_dxb	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
5G加包模型	5gbyw	kehujingyingbudb.ml_xy_yw_m_new	s_yw_team	con_yw_team	 sig_offer_spec_name_teamzk not like '%5g%' and pack_5g_internet_num_teamadd=0 and is_5g_main_offer_teamzk='0' and is_5g_upgrade_offer_teamzk='0'	available_notzd	acct_month	zk_user_no	flag_5gb	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
视频彩铃潜客模型	clyw	kehujingyingbudb.ml_xy_yw_m_new	s_yw_team	con_yw_team	 pack_spcl_num_teamadd=0	available_notzd	acct_month	zk_user_no	flag_cl	1	0	1	np.nan	['acct_month', 'area_no_teamzk']	{'arpu_teamadd': False, 'data_flux_teamadd': False}	Ncase
"""

# </editor-fold> -------------------------------------------------------------------------------------------------------


# --------------------------------- <editor-fold desc="数据库基础表信息"> ----------------------------------------------
# 从excel粘贴过来的数据库基础表信息，用于./selfmodule/tablemodule/tablefun.py中的privy_basedatafun()、tab_explore_create()等
s_table_info = """s_field_base	tabletype	tablename	alias	on	tableXday_desc	tableXscore_desc
s_yw_fake	tableXmain	kehujingyingbudb.ml_feature_info_yw_user_m	x0	主表		
s_yw_fake	tableXadd	kehujingyingbudb.ml_feature_add_yw_user_m	x1	x0.user_id = x1.user_id		
s_yw_fake	tableXday	kehujingyingbudb.ml_feature_info_yw_user_day	d1	x0.user_id = d1.user_id	{"col_day": 'acct_day', 'monthadd': 1, 'dd': 15,'day_datetye':2}	
s_yw_fake	tableXscore	kehujingyingbudb.table_score_month	s	x0.user_id = s.user_id		{'if_unpivot': True}
s_yw_fake	tableY	kehujingyingbudb.ml_target_info_yw_user_m	y	x0.user_id = y.user_id		
s_yw_team	tableXmain	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	x0	主表		
s_yw_team	tableXadd	kehujingyingbudb.dm_zc_yw_moxing_team_add_m	x1	x0.zk_user_no = x1.zk_user_no		
s_yw_team	tableXscore	kehujingyingbudb.binaryclassify_score_m	s	x0.zk_user_no = s.user_no		{'if_unpivot': True}
s_yw_team	tableXday	edww.dww_d_pr_pri_al_inst	d1	x0.zk_user_no = d1.user_no	{"col_day":['acct_month', 'day_id'], 'monthadd': 1, 'dd': 10,'day_datetye':1}	
s_yw_team	tableY	kehujingyingbudb.dm_zc_yw_moxing_team_target_m	y	x0.zk_user_no = y.zk_user_no		
s_kd_user	tableXmain	kehujingyingbudb.dm_zc_kd_moxing_info_m	x0	主表		
s_kd_user	tableY	kehujingyingbudb.dm_zc_kd_moxing_target_m	y	x0.user_no = y.user_no		
s_kd_user	tableXday	edww.dww_d_pr_pri_al_inst	d1	x0.user_no = d1.user_no	{"col_day":['acct_month', 'day_id'], 'monthadd': 1, 'dd': 15,'day_datetye':1}	
s_rh_team	tableXmain	kehujingyingbudb.dm_zc_rh_moxing_team_info_m	x0	主表		
s_rh_team	tableXadd	kehujingyingbudb.dm_zc_yw_moxing_team_add_m	x1	x0.zk_user_no = x1.zk_user_no		
s_rh_team	tableXscore	kehujingyingbudb.binaryclassify_score_m	s	x0.zk_user_no = s.user_no		
s_rh_team	tableY	kehujingyingbudb.dm_zc_yw_moxing_team_target_m	y	x0.zk_user_no = y.zk_user_no		
s_rh_team	tableXday	edww.dww_d_pr_pri_al_inst	d1	x0.zk_user_no = d1.user_no	{"col_day":['acct_month', 'day_id'], 'monthadd': 1, 'dd': 15,'day_datetye':1}	
s_yw_user	tableXmain	kehujingyingbudb.dm_zc_yw_moxing_info_m	x0	主表		
s_yw_user	tableXscore	kehujingyingbudb.binaryclassify_score_m	s	x0.user_no = s.user_no		
s_yw_user	tableY	kehujingyingbudb.dm_zc_yw_moxing_target_m	y	x0.user_no = y.user_no				
"""
# </editor-fold> -------------------------------------------------------------------------------------------------------

#
#
# ----------------------------- <editor-fold desc="移网基础宽表_用户to套餐组"> -----------------------------------------
s_yw_toteam = """原始字段名	数据类型	字段说明	汇总方式	后缀
month_id	varchar(6)	账期		
user_no	varchar(20)	用户ID	_组内主卡取值	_teamzk
acct_id	varchar(20)	账户ID	_组内主卡取值	_teamzk
customer_no	varchar(30)	客户ID	_组内主卡取值	_teamzk
dev_channel_nbr	varchar(30)	入网渠道ID	_组内主卡取值	_teamzk
dev_channel_name	varchar(500)	入网渠道名称	_组内主卡取值	_teamzk
open_date	varchar(14)	入网日期	_组内主卡取值	_teamzk
open_months	int	入网时长(月)	_组内主卡取值	_teamzk
cert_type	varchar(1)	证件类型(1:个人,0:非个人)	_组内主卡取值	_teamzk
cert_nbr	varchar(100)	身份证号码	_组内主卡取值	_teamzk
area_no	varchar(5)	地市名称	_组内主卡取值	_teamzk
city_no	varchar(20)	区县名称	_组内主卡取值	_teamzk
is_village	varchar(1)	农村标识	_组内主卡取值	_teamzk
age	varchar(3)	年龄	_组内主卡取值	_teamzk
sex	varchar(2)	性别	_组内主卡取值	_teamzk
is_valid	varchar(1)	是否在网	_组内主卡取值	_teamzk
std_user_status_name	varchar(250)	用户状态名称	_组内主卡取值	_teamzk
stop_times	int	停机次数	_组内总和	_teamadd
resent_stop_date	varchar(14)	最近一次停机日期	_组内最大值	_teammax
is_acct	varchar(1)	是否出账	_组内主卡取值	_teamzk
payment_mode_cd	varchar(10)	付费方式	_组内主卡取值	_teamzk
is_real_name	varchar(1)	是否实名用户	_组内主卡取值	_teamzk
is_gz	varchar(1)	是否公众用户	_组内主卡取值	_teamzk
is_xy	varchar(1)	是否校园用户	_组内主卡取值	_teamzk
is_gov	varchar(1)	是否政企用户	_组内主卡取值	_teamzk
is_grp_member	varchar(1)	是否集团成员	_组内主卡取值	_teamzk
is_hlwk	varchar(1)	是否互联网卡	_组内主卡取值	_teamzk
is_sw	varchar(1)	是否三无用户	_组内主卡取值	_teamzk
is_red_list	varchar(1)	剔除口径类字段	_组内主卡取值	_teamzk
is_5g_main_offer	varchar(1)	是否5G主销售品用户	_组内主卡取值	_teamzk
is_5g_upgrade_offer	varchar(1)	是否5G流量包用户	_组内主卡取值	_teamzk
is_fq	varchar(1)	是否分期用户	_组内主卡取值	_teamzk
is_usim	varchar(1)	是否usim卡	_组内主卡取值	_teamzk
tyaddresscode	varchar(255)	小区编码	_组内主卡取值	_teamzk
tyaddressname	varchar(255)	小区名称	_组内主卡取值	_teamzk
shentoulv	varchar(10)	小区渗透率	_组内主卡取值	_teamzk
family_form_ersj	varchar(2)	家庭构成_二人世界	_组内主卡取值	_teamzk
family_form_skzj	varchar(2)	家庭构成_三口之家	_组内主卡取值	_teamzk
family_form_unknown	varchar(2)	家庭构成_未知	_组内主卡取值	_teamzk
family_prop_child	varchar(2)	家庭特征_家有儿童	_组内主卡取值	_teamzk
family_prop_elder	varchar(2)	家庭特征_家有老人	_组内主卡取值	_teamzk
family_prop_unknown	varchar(2)	家庭特征_未知	_组内主卡取值	_teamzk
family_type	varchar(100)	家庭类型	_组内主卡取值	_teamzk
sig_offer_spec_id	varchar(20)	单产品套餐ID	_组内主卡取值	_teamzk
sig_offer_spec_name	varchar(200)	单产品套餐名称	_组内主卡取值	_teamzk
sig_offer_create_dt	varchar(14)	单产品套餐订购时间	_组内主卡取值	_teamzk
comp_offer_spec_id	varchar(20)	融合套餐ID	_组内主卡取值	_teamzk
comp_offer_spec_name	varchar(200)	融合套餐名称	_组内主卡取值	_teamzk
comp_offer_create_dt	varchar(14)	融合套餐订购时间	_组内主卡取值	_teamzk
is_dupety	varchar(1)	主副卡类型(0:主卡,1:副卡)	_组内主卡取值	_teamzk
fk_num	int	副卡数量	_组内主卡取值	_teamzk
zk_user_no	varchar(20)	主卡ID		
prod_spec_fee	decimal(20,4)	套餐月费	_组内主卡取值	_teamzk
prod_spec_fluw	decimal(20,4)	套内流量	_组内主卡取值	_teamzk
prod_spec_dur	decimal(20,4)	套内语音	_组内主卡取值	_teamzk
mbl_inner_fluw	decimal(20,4)	套内流量使用量	_组内总和	_teamadd
dialing_inner	decimal(20,4)	套内语音使用量	_组内总和	_teamadd
is_comp	varchar(1)	是否融合业务	_组内主卡取值	_teamzk
comp_user_no	varchar(50)	融合组ID	_组内主卡取值	_teamzk
comp_eff_date	varchar(14)	融合业务生效时间	_组内主卡取值	_teamzk
comp_exp_date	varchar(14)	融合业务失效时间	_组内主卡取值	_teamzk
comp_type	varchar(20)	融合产品结构	_组内主卡取值	_teamzk
comp_num	int	融合产品数	_组内主卡取值	_teamzk
comp_kd_num	int	融合宽带数	_组内主卡取值	_teamzk
comp_yd_num	int	融合移动数	_组内主卡取值	_teamzk
comp_hd_num	int	融合电视数	_组内主卡取值	_teamzk
comp_wf_num	int	融合WiFi数	_组内主卡取值	_teamzk
comp_kj_num	int	融合看家数	_组内主卡取值	_teamzk
is_agre	varchar(1)	是否有合约	_组内主卡取值	_teamzk
track_eff_date	varchar(14)	合约生效时间	_组内主卡取值	_teamzk
track_exp_date	varchar(14)	合约失效时间	_组内主卡取值	_teamzk
pack_hd_num	int	订购电视包数	_组内总和	_teamadd
pack_dur_num	int	订购语音包数	_组内总和	_teamadd
pack_flow_num	int	订购流量包数	_组内总和	_teamadd
pack_flow_sum	decimal(20,4)	订购包总流量	_组内总和	_teamadd
pack_dur_sum	decimal(20,4)	订购包总语音	_组内总和	_teamadd
pack_month_flow_num	int	订购流量月包数	_组内总和	_teamadd
pack_month_flow_sum	decimal(20,4)	订购流量月包总流量	_组内总和	_teamadd
pack_month_flow_num_m	int	当月订购流量月包数	_组内总和	_teamadd
pack_month_flow_sum_m	decimal(20,4)	当月订购流量月包总流量	_组内总和	_teamadd
pack_month_flow_exp_date	varchar(14)	办理流量月包失效时间	_组内主卡取值	_teamzk
pack_directed_num	int	订购定向流量包数	_组内总和	_teamadd
pack_directed_sum	decimal(20,4)	订购定向流量包总流量	_组内总和	_teamadd
pack_directed_num_m	int	当月订购定向流量包数	_组内总和	_teamadd
pack_directed_sum_m	decimal(20,4)	当月订购定向流量包总流量	_组内总和	_teamadd
pack_directed_exp_date	varchar(14)	办理定向流量包失效时间	_组内主卡取值	_teamzk
pack_5g_internet_num	int	订购5G网络包数	_组内总和	_teamadd
pack_5g_internet_free_num	int	订购0元5G网络包数	_组内总和	_teamadd
pack_5g_internet_sum	decimal(20,4)	订购5G网络包总流量	_组内总和	_teamadd
pack_5g_internet_num_m	int	当月订购5G网络包总数	_组内总和	_teamadd
pack_5g_internet_free_num_m	int	当月订购0元5G网络包总数	_组内总和	_teamadd
pack_5g_internet_sum_m	decimal(20,4)	当月订购5G网络包总流量	_组内总和	_teamadd
pack_5g_internet_exp_date	varchar(14)	办理5G包失效时间	_组内主卡取值	_teamzk
pack_spcl_num	int	订购视频彩铃包数	_组内总和	_teamadd
pack_spcl_free_num	int	订购0元视频彩铃包数	_组内总和	_teamadd
pack_spcl_num_m	int	当月订购视频彩铃包数	_组内总和	_teamadd
pack_spcl_free_num_m	int	当月订购0元视频彩铃包数	_组内总和	_teamadd
pack_spcl_exp_date	varchar(14)	办理视频彩铃失效时间	_组内主卡取值	_teamzk
pack_spcl_6_month	varchar(2)	近半年是否办理过视频彩铃	_组内最大值	_teammax
is_hy_or_hb	varchar(1)	是否订购合约及红包类销售品	_组内主卡取值	_teamzk
point	decimal(20,4)	积分	_组内主卡取值	_teamzk
star	decimal(20,4)	星级	_组内主卡取值	_teamzk
basic_credit	decimal(20,4)	信用度	_组内主卡取值	_teamzk
quota	decimal(20,4)	授信额度	_组内主卡取值	_teamzk
is_lh	varchar(1)	是否靓号	_组内主卡取值	_teamzk
lh_type	varchar(20)	靓号类型	_组内主卡取值	_teamzk
is_bxl	varchar(1)	当前是否不限量	_组内主卡取值	_teamzk
bxl_flow_step	varchar(20)	不限量流量档位	_组内主卡取值	_teamzk
bxl_deal_times	int	不限量包的办理次数	_组内主卡取值	_teamzk
is_deal_bxl	varchar(1)	是否曾办理过不限量包	_组内主卡取值	_teamzk
pack_bxl_num	int	当前不限量包数量	_组内主卡取值	_teamzk
owe_flag	varchar(1)	当前是否欠费	_组内主卡取值	_teamzk
owe_charge	decimal(20,4)	当前欠费金额	_组内总和	_teamadd
owe_times	int	欠费次数	_组内总和	_teamadd
last_owe_month	varchar(6)	最近一次欠费时间	_组内最大值	_teammax
last_owe_charge	decimal(20,4)	最近一次欠费金额	_组内主卡取值	_teamzk
owe_charge_acct	decimal(20,4)	当月产生欠费总金额	_组内总和	_teamadd
payment_cnt	decimal(20,4)	缴费次数	_组内总和	_teamadd
payment_charge	decimal(20,4)	缴费金额	_组内总和	_teamadd
payment_time	varchar(14)	最近一次缴费时间	_组内最大值	_teammax
payment_fee	decimal(20,4)	最近一次缴费金额	_组内主卡取值	_teamzk
balance	decimal(20,4)	余额	_组内总和	_teamadd
arpu	decimal(20,4)	ARPU	_组内总和	_teamadd
call_fee	decimal(20,4)	通话费	_组内总和	_teamadd
call_local_fee	decimal(20,4)	本地通话费	_组内总和	_teamadd
call_long_prov_fee	decimal(20,4)	长途通话费	_组内总和	_teamadd
call_roam_fee	decimal(20,4)	漫游通话费	_组内总和	_teamadd
flux_fee	decimal(20,4)	流量费	_组内总和	_teamadd
sms_fee	decimal(20,4)	短信费	_组内总和	_teamadd
rgst_tmn_no	varchar(10)	终端IMEI	_组内主卡取值	_teamzk
rgst_tmn_brand	varchar(50)	终端品牌	_组内主卡取值	_teamzk
rgst_tmn_model	varchar(32)	终端型号	_组内主卡取值	_teamzk
rgst_tmn_type	varchar(100)	终端类型	_组内主卡取值	_teamzk
rgst_tmn_time	varchar(20)	注册日期	_组内主卡取值	_teamzk
rgst_tmn_flag	varchar(20)	是否智能机(1:是 0:否)	_组内主卡取值	_teamzk
phone_price	varchar(50)	价格	_组内主卡取值	_teamzk
operating_sys	varchar(255)	操作系统	_组内主卡取值	_teamzk
market_time	varchar(255)	上市时间	_组内主卡取值	_teamzk
prd_position	varchar(200)	产品定位	_组内主卡取值	_teamzk
screens_nbr	varchar(200)	屏幕数量	_组内主卡取值	_teamzk
main_screen_size	varchar(200)	主屏幕尺寸	_组内主卡取值	_teamzk
baseband_chip_clocked	varchar(100)	基带芯片主频	_组内主卡取值	_teamzk
resolution	varchar(200)	显示分辩率	_组内主卡取值	_teamzk
ram	varchar(200)	RAM	_组内主卡取值	_teamzk
main_camera	varchar(200)	主摄像头	_组内主卡取值	_teamzk
start_use_time	varchar(8)	开始使用时间	_组内主卡取值	_teamzk
end_use_time	varchar(8)	结束使用时间	_组内主卡取值	_teamzk
usage_days	int	当次使用时长(天)	_组内主卡取值	_teamzk
usage_days_sum	int	总计使用时长(天)	_组内主卡取值	_teamzk
is_in_use	varchar(1)	当前是否正在使用	_组内主卡取值	_teamzk
is_new_tmn	varchar(1)	是否为新终端（按品牌）	_组内主卡取值	_teamzk
rgst_tmn_brand_1	varchar(50)	上一个终端_终端品牌	_组内主卡取值	_teamzk
rgst_tmn_model_1	varchar(32)	上一个终端_终端型号	_组内主卡取值	_teamzk
rgst_tmn_type_1	varchar(100)	上一个终端_终端类型	_组内主卡取值	_teamzk
rgst_tmn_time_1	varchar(20)	上一个终端_注册日期	_组内主卡取值	_teamzk
rgst_tmn_flag_1	varchar(20)	上一个终端_是否智能机(1:是 0:否)	_组内主卡取值	_teamzk
phone_price_1	varchar(50)	上一个终端_价格	_组内主卡取值	_teamzk
operating_sys_1	varchar(255)	上一个终端_操作系统	_组内主卡取值	_teamzk
market_time_1	varchar(255)	上一个终端_上市时间	_组内主卡取值	_teamzk
prd_position_1	varchar(200)	上一个终端_产品定位	_组内主卡取值	_teamzk
screens_nbr_1	varchar(200)	上一个终端_屏幕数量	_组内主卡取值	_teamzk
main_screen_size_1	varchar(200)	上一个终端_主屏幕尺寸	_组内主卡取值	_teamzk
baseband_chip_clocked_1	varchar(100)	上一个终端_基带芯片主频	_组内主卡取值	_teamzk
resolution_1	varchar(200)	上一个终端_显示分辩率	_组内主卡取值	_teamzk
ram_1	varchar(200)	上一个终端_RAM	_组内主卡取值	_teamzk
main_camera_1	varchar(200)	上一个终端_主摄像头	_组内主卡取值	_teamzk
start_use_time_1	varchar(8)	上一个终端_开始使用时间	_组内主卡取值	_teamzk
end_use_time_1	varchar(8)	上一个终端_结束使用时间	_组内主卡取值	_teamzk
usage_days_1	int	上一个终端_当次使用时长(天)	_组内主卡取值	_teamzk
usage_days_sum_1	int	上一个终端_总计使用时长(天)	_组内主卡取值	_teamzk
is_new_tmn_1	varchar(1)	上一个终端_是否为新终端（按品牌）	_组内主卡取值	_teamzk
rgst_tmn_brand_2	varchar(50)	上两个终端_终端品牌	_组内主卡取值	_teamzk
rgst_tmn_model_2	varchar(32)	上两个终端_终端型号	_组内主卡取值	_teamzk
rgst_tmn_type_2	varchar(100)	上两个终端_终端类型	_组内主卡取值	_teamzk
rgst_tmn_time_2	varchar(20)	上两个终端_注册日期	_组内主卡取值	_teamzk
rgst_tmn_flag_2	varchar(20)	上两个终端_是否智能机(1:是 0:否)	_组内主卡取值	_teamzk
phone_price_2	varchar(50)	上两个终端_价格	_组内主卡取值	_teamzk
operating_sys_2	varchar(255)	上两个终端_操作系统	_组内主卡取值	_teamzk
market_time_2	varchar(255)	上两个终端_上市时间	_组内主卡取值	_teamzk
prd_position_2	varchar(200)	上两个终端_产品定位	_组内主卡取值	_teamzk
screens_nbr_2	varchar(200)	上两个终端_屏幕数量	_组内主卡取值	_teamzk
main_screen_size_2	varchar(200)	上两个终端_主屏幕尺寸	_组内主卡取值	_teamzk
baseband_chip_clocked_2	varchar(100)	上两个终端_基带芯片主频	_组内主卡取值	_teamzk
resolution_2	varchar(200)	上两个终端_显示分辩率	_组内主卡取值	_teamzk
ram_2	varchar(200)	上两个终端_RAM	_组内主卡取值	_teamzk
main_camera_2	varchar(200)	上两个终端_主摄像头	_组内主卡取值	_teamzk
start_use_time_2	varchar(8)	上两个终端_开始使用时间	_组内主卡取值	_teamzk
end_use_time_2	varchar(8)	上两个终端_结束使用时间	_组内主卡取值	_teamzk
usage_days_2	int	上两个终端_当次使用时长(天)	_组内主卡取值	_teamzk
usage_days_sum_2	int	上两个终端_总计使用时长(天)	_组内主卡取值	_teamzk
is_new_tmn_2	varchar(1)	上两个终端_是否为新终端（按品牌）	_组内主卡取值	_teamzk
is_shaungka	varchar(1)	是否双卡终端	_组内主卡取值	_teamzk
shaungka_type	varchar(10)	双卡终端的卡类型	_组内主卡取值	_teamzk
dur	decimal(20,4)	通话时长	_组内总和	_teamadd
calling_dur	decimal(20,4)	主叫时长	_组内总和	_teamadd
called_dur	decimal(20,4)	被叫时长	_组内总和	_teamadd
cnt	int	通话次数	_组内总和	_teamadd
calling_cnt	int	主叫次数	_组内总和	_teamadd
called_cnt	int	被叫次数	_组内总和	_teamadd
roam_dur	decimal(20,4)	漫游通话时长	_组内总和	_teamadd
roam_cnt	decimal(20,4)	漫游通话次数	_组内总和	_teamadd
roam_out_dur	decimal(20,4)	国际漫游通话时长（含港澳台）	_组内总和	_teamadd
roam_dmstc_dur	decimal(20,4)	省外漫游通话时长（国内且省外）	_组内总和	_teamadd
local_dur	decimal(20,4)	市内通话时长	_组内总和	_teamadd
call_days	int	通话天数	_组内最大值	_teammax
calling_days	int	主叫天数	_组内最大值	_teammax
called_days	int	被叫天数	_组内最大值	_teammax
roam_out_days	int	国际漫游通话天数（含港澳台）	_组内最大值	_teammax
roam_dmstc_days	int	省外漫游通话天数（国内且省外）	_组内最大值	_teammax
local_days	int	市内通话天数	_组内最大值	_teammax
ct_dur	decimal(20,4)	网内通话时长	_组内总和	_teamadd
wj_dur	decimal(20,4)	网间通话时长	_组内总和	_teamadd
ct_calling_dur	decimal(20,4)	网内主叫时长	_组内总和	_teamadd
ct_cnt	decimal(20,4)	网内通话次数	_组内总和	_teamadd
ct_calling_cnt	decimal(20,4)	网内主叫次数	_组内总和	_teamadd
wj_calling_dur	decimal(20,4)	网间主叫时长	_组内总和	_teamadd
wj_cnt	decimal(20,4)	网间通话次数	_组内总和	_teamadd
wj_calling_cnt	decimal(20,4)	网间主叫次数	_组内总和	_teamadd
jwq_num	int	通话交往圈用户数	_组内总和	_teamadd
jwq_ct_num	int	通话交往圈网内用户数	_组内总和	_teamadd
jwq_wj_num	int	通话交往圈网间用户数	_组内总和	_teamadd
jwq_calling_num	int	主叫交往圈用户数	_组内总和	_teamadd
jwq_ct_calling_num	int	主叫交往圈网内用户数	_组内总和	_teamadd
jwq_wj_calling_num	int	主叫交往圈网间用户数	_组内总和	_teamadd
jwq_called_num	int	被叫交往圈用户数	_组内总和	_teamadd
jwq_ct_called_num	int	被叫交往圈网内用户数	_组内总和	_teamadd
jwq_wj_called_num	int	被叫交往圈网间用户数	_组内总和	_teamadd
data_flux	decimal(20,4)	流量	_组内总和	_teamadd
last_to_acct_flux	decimal(20,4)	上月递延本月流量	_组内主卡取值	_teamzk
acct_to_next_flux	decimal(20,4)	本月递延下月流量	_组内总和	_teamadd
flux_5g	decimal(20,4)	5G流量	_组内总和	_teamadd
flux_4g	decimal(20,4)	4G流量	_组内总和	_teamadd
flux_3g	decimal(20,4)	3G流量	_组内总和	_teamadd
flux_2g	decimal(20,4)	2G流量	_组内总和	_teamadd
int_flux	decimal(20,4)	国际漫游流量（含港澳台）	_组内总和	_teamadd
dmstc_flux	decimal(20,4)	省外漫游流量（国内且省外）	_组内总和	_teamadd
flux_days	int	流量使用天数	_组内最大值	_teammax
int_flux_days	int	国际漫游流量使用天数（含港澳台）	_组内最大值	_teammax
dmstc_flux_days	int	省外漫游流量使用天数（国内且省外）	_组内最大值	_teammax
normal_flux	decimal(20,4)	日间流量	_组内总和	_teamadd
normal_flux_days	int	日间流量使用天数	_组内最大值	_teammax
night_flux	decimal(20,4)	夜间流量	_组内总和	_teamadd
night_flux_days	int	夜间流量使用天数	_组内最大值	_teammax
weekday_flux_avg	decimal(20,4)	工作日日均流量	_组内总和	_teamadd
weekend_flux_avg	decimal(20,4)	节假日日均流量	_组内总和	_teamadd
sx_flux_avg	decimal(20,4)	上旬日均流量	_组内总和	_teamadd
zx_flux_avg	decimal(20,4)	中旬日均流量	_组内总和	_teamadd
xx_flux_avg	decimal(20,4)	下旬日均流量	_组内总和	_teamadd
mbl_out_flow	decimal(20,5)	溢出流量	_组内总和	_teamadd
sms_cnt	int	短信次数	_组内总和	_teamadd
sms_send_cnt	int	短信上行次数	_组内总和	_teamadd
sms_recv_cnt	int	短信下行次数	_组内总和	_teamadd
self_scs_cnt	int	本网客服--咨询次数	_组内总和	_teamadd
self_scs_tsbz_cnt	int	本网客服--投诉报障次数	_组内总和	_teamadd
call_turn_cnt	int	呼转次数	_组内总和	_teamadd
"""
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# ------------------------------- <editor-fold desc="移网field_base_用户"> -------------------------------------------
s_yw_user = """field_name	comment	dtype_db	dtype_classify	field_src	table	available	available_notzd	available_zd	formula	remark	must_remain	into_model	is_cause
month_id	账期	varchar(6)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m							删除	否
user_no	用户ID_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
acct_id	账户ID_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
customer_no	客户ID_组内主卡取值	varchar(30)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
dev_channel_nbr	入网渠道ID_组内主卡取值	varchar(30)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	否
dev_channel_name	入网渠道名称_组内主卡取值	varchar(500)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生		删除	否
open_date	入网日期_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
open_months	入网时长(月)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生	是		
cert_type	证件类型(1:个人,0:非个人)_组内主卡取值	varchar(1)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
cert_nbr	身份证号码_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	否
area_no	地市名称_组内主卡取值	varchar(5)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			否
city_no	区县名称_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			否
is_village	农村标识_组内主卡取值	varchar(1)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
age	年龄_组内主卡取值	varchar(3)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生			
sex	性别_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生			否
is_valid	是否在网_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			否
std_user_status_name	用户状态名称_组内主卡取值	varchar(250)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
stop_times	停机次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
resent_stop_date	最近一次停机日期_组内最大值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
monthsaready_resent_stop_date	最近一次停机日期_组内最大值_已发生时长		数值型	手动衍生_py	python				current_date - resent_stop_date	不参与近n月自动衍生			
is_acct	是否出账_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
payment_mode_cd	付费方式_组内主卡取值	varchar(10)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
is_real_name	是否实名用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生			
is_gz	是否公众用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_xy	是否校园用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_gov	是否政企用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_grp_member	是否集团成员_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_hlwk	是否互联网卡_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_sw	是否三无用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_red_list	剔除口径类字段_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_5g_main_offer	是否5G主销售品用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
is_5g_upgrade_offer	是否5G流量包用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
is_fq	是否分期用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
is_usim	是否usim卡_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
tyaddresscode	小区编码_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	
tyaddressname	小区名称_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	
shentoulv	小区渗透率_组内主卡取值	varchar(10)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
family_form_ersj	家庭构成_二人世界_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生	是		
family_form_skzj	家庭构成_三口之家_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生	是		
family_form_unknown	家庭构成_未知_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生	是		
family_prop_child	家庭特征_家有儿童_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生	是		
family_prop_elder	家庭特征_家有老人_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生	是		
family_prop_unknown	家庭特征_未知_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生	是		
family_type	家庭类型_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生	是		
sig_offer_spec_id	单产品套餐ID_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	否
sig_offer_spec_name	单产品套餐名称_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	
sig_offer_create_dt	单产品套餐订购时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
monthsaready_sig_offer_create_dt	单产品套餐订购时间_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - sig_offer_create_dt	不参与近n月自动衍生			
comp_offer_spec_id	融合套餐ID_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	否
comp_offer_spec_name	融合套餐名称_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	
comp_offer_create_dt	融合套餐订购时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
monthsaready_comp_offer_create_dt	融合套餐订购时间_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - comp_offer_create_dt	不参与近n月自动衍生			
is_dupety	主副卡类型(0:主卡,1:副卡)_组内主卡取值	varchar(1)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
fk_num	副卡数量_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
zk_user_no	主卡ID	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生			否
prod_spec_fee	套餐月费_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
prod_spec_fluw	套内流量_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
prod_spec_dur	套内语音_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
mbl_inner_fluw	套内流量使用量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
dialing_inner	套内语音使用量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
is_comp	是否融合业务_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
comp_user_no	融合组ID_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
comp_eff_date	融合业务生效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
monthsaready_comp_eff_date	融合业务生效时间_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - comp_eff_date	不参与近n月自动衍生			
comp_exp_date	融合业务失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
monthsremain_comp_exp_date	融合业务失效时间_组内主卡取值_剩余时长		数值型	手动衍生_py	python				comp_exp_date - current_date	不参与近n月自动衍生			
comp_type	融合产品结构_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
comp_num	融合产品数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
comp_kd_num	融合宽带数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
comp_yd_num	融合移动数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
comp_hd_num	融合电视数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
comp_wf_num	融合WiFi数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
comp_kj_num	融合看家数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
is_agre	是否有合约_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
track_eff_date	合约生效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
monthsaready_track_eff_date	合约生效时间_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - track_eff_date	不参与近n月自动衍生			
track_exp_date	合约失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
monthsremain_track_exp_date	合约失效时间_组内主卡取值_剩余时长		数值型	手动衍生_py	python				track_exp_date - current_date	不参与近n月自动衍生			
pack_hd_num	订购电视包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
pack_dur_num	订购语音包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
pack_flow_num	订购流量包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
pack_flow_sum	订购包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
pack_dur_sum	订购包总语音_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
pack_month_flow_num	订购流量月包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
pack_month_flow_sum	订购流量月包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
pack_month_flow_num_m	当月订购流量月包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用					是		
pack_month_flow_sum_m	当月订购流量月包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用					是		
pack_month_flow_exp_date	办理流量月包失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
pack_directed_num	订购定向流量包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
pack_directed_sum	订购定向流量包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
pack_directed_num_m	当月订购定向流量包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
pack_directed_sum_m	当月订购定向流量包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
pack_directed_exp_date	办理定向流量包失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
pack_5g_internet_num	订购5G网络包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
pack_5g_internet_free_num	订购0元5G网络包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
pack_5g_internet_sum	订购5G网络包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
pack_5g_internet_num_m	当月订购5G网络包总数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
pack_5g_internet_free_num_m	当月订购0元5G网络包总数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生			
pack_5g_internet_sum_m	当月订购5G网络包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
pack_5g_internet_exp_date	办理5G包失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
pack_spcl_num	订购视频彩铃包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
pack_spcl_free_num	订购0元视频彩铃包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
pack_spcl_num_m	当月订购视频彩铃包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
pack_spcl_free_num_m	当月订购0元视频彩铃包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生			
pack_spcl_exp_date	办理视频彩铃失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
monthsremain_pack_spcl_exp_date	办理视频彩铃失效时间_组内主卡取值_剩余时长		数值型	手动衍生_py	python				pack_spcl_exp_date - current_date	不参与近n月自动衍生			
pack_spcl_6_month	近半年是否办理过视频彩铃_组内最大值	varchar(2)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生			
is_hy_or_hb	是否订购合约及红包类销售品_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
point	积分_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
star	星级_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
basic_credit	信用度_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
quota	授信额度_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
is_lh	是否靓号_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
lh_type	靓号类型_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
is_bxl	当前是否不限量_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用					是		
bxl_flow_step	不限量流量档位_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
bxl_deal_times	不限量包的办理次数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
is_deal_bxl	是否曾办理过不限量包_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
pack_bxl_num	当前不限量包数量_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
owe_flag	当前是否欠费_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
owe_charge	当前欠费金额_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
owe_times	欠费次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
last_owe_month	最近一次欠费时间_组内最大值	varchar(6)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
monthsaready_last_owe_month	最近一次欠费时间_组内最大值_已发生时长		数值型	手动衍生_py	python				current_date - last_owe_month	不参与近n月自动衍生			
last_owe_charge	最近一次欠费金额_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生			
owe_charge_acct	当月产生欠费总金额_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
payment_cnt	缴费次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
payment_charge	缴费金额_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
payment_time	最近一次缴费时间_组内最大值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
monthsaready_payment_time	最近一次缴费时间_组内最大值_已发生时长		数值型	手动衍生_py	python				current_date - payment_time	不参与近n月自动衍生			
payment_fee	最近一次缴费金额_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与近n月自动衍生			
balance	余额_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用					是		
arpu	ARPU_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
arpu_p	arpu/余额		数值型	手动衍生_py	python				arpu / balance		是		
call_fee	通话费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
call_local_fee	本地通话费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
call_long_prov_fee	长途通话费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
call_roam_fee	漫游通话费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
flux_fee	流量费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用					是		
sms_fee	短信费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
rgst_tmn_no	终端IMEI_组内主卡取值	varchar(10)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	
rgst_tmn_brand	终端品牌_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
rgst_tmn_model	终端型号_组内主卡取值	varchar(32)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生		删除	
rgst_tmn_type	终端类型_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
rgst_tmn_time	注册日期_组内主卡取值	varchar(20)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
monthsaready_rgst_tmn_time	注册日期_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - rgst_tmn_time	不参与近n月自动衍生			
rgst_tmn_flag	是否智能机(1:是 0:否)_组内主卡取值	varchar(20)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
phone_price	价格_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
operating_sys	操作系统_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
market_time	上市时间_组内主卡取值	varchar(255)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
prd_position	产品定位_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
screens_nbr	屏幕数量_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
main_screen_size	主屏幕尺寸_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
baseband_chip_clocked	基带芯片主频_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
resolution	显示分辩率_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
ram	RAM_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用	不可用			不参与近n月自动衍生			
main_camera	主摄像头_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用	不可用			不参与近n月自动衍生			
start_use_time	开始使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
end_use_time	结束使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
usage_days	当次使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
usage_days_sum	总计使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_in_use	当前是否正在使用_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_new_tmn	是否为新终端（按品牌）_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
rgst_tmn_brand_1	上一个终端_终端品牌_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_model_1	上一个终端_终端型号_组内主卡取值	varchar(32)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生		删除	
rgst_tmn_type_1	上一个终端_终端类型_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_time_1	上一个终端_注册日期_组内主卡取值	varchar(20)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
rgst_tmn_flag_1	上一个终端_是否智能机(1:是 0:否)_组内主卡取值	varchar(20)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
phone_price_1	上一个终端_价格_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
operating_sys_1	上一个终端_操作系统_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
market_time_1	上一个终端_上市时间_组内主卡取值	varchar(255)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
prd_position_1	上一个终端_产品定位_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
screens_nbr_1	上一个终端_屏幕数量_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
main_screen_size_1	上一个终端_主屏幕尺寸_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
baseband_chip_clocked_1	上一个终端_基带芯片主频_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
resolution_1	上一个终端_显示分辩率_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
ram_1	上一个终端_RAM_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用	不可用			不参与近n月自动衍生			
main_camera_1	上一个终端_主摄像头_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用	不可用			不参与近n月自动衍生			
start_use_time_1	上一个终端_开始使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
end_use_time_1	上一个终端_结束使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
usage_days_1	上一个终端_当次使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
usage_days_sum_1	上一个终端_总计使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_new_tmn_1	上一个终端_是否为新终端（按品牌）_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_brand_2	上两个终端_终端品牌_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_model_2	上两个终端_终端型号_组内主卡取值	varchar(32)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生		删除	
rgst_tmn_type_2	上两个终端_终端类型_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_time_2	上两个终端_注册日期_组内主卡取值	varchar(20)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
rgst_tmn_flag_2	上两个终端_是否智能机(1:是 0:否)_组内主卡取值	varchar(20)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
phone_price_2	上两个终端_价格_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
operating_sys_2	上两个终端_操作系统_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
market_time_2	上两个终端_上市时间_组内主卡取值	varchar(255)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
prd_position_2	上两个终端_产品定位_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
screens_nbr_2	上两个终端_屏幕数量_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
main_screen_size_2	上两个终端_主屏幕尺寸_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
baseband_chip_clocked_2	上两个终端_基带芯片主频_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
resolution_2	上两个终端_显示分辩率_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
ram_2	上两个终端_RAM_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用	不可用			不参与近n月自动衍生			
main_camera_2	上两个终端_主摄像头_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用	不可用			不参与近n月自动衍生			
start_use_time_2	上两个终端_开始使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
end_use_time_2	上两个终端_结束使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与自动衍生		删除	否
usage_days_2	上两个终端_当次使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
usage_days_sum_2	上两个终端_总计使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
is_new_tmn_2	上两个终端_是否为新终端（按品牌）_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m		不可用			不参与近n月自动衍生			
is_shaungka	是否双卡终端_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
shaungka_type	双卡终端的卡类型_组内主卡取值	varchar(10)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与近n月自动衍生			
dur	通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
calling_dur	主叫时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
called_dur	被叫时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
cnt	通话次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
calling_cnt	主叫次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
called_cnt	被叫次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
roam_dur	漫游通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
roam_cnt	漫游通话次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
roam_out_dur	国际漫游通话时长（含港澳台）_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
roam_dmstc_dur	省外漫游通话时长（国内且省外）_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
local_dur	市内通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
call_days	通话天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
call_days_p	通话天数_组内最大值/当月天数		数值型	手动衍生_py	python				call_days / days_month				
calling_days	主叫天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
calling_days_p	主叫天数_组内最大值/当月天数		数值型	手动衍生_py	python				calling_days / days_month				
called_days	被叫天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
called_days_p	被叫天数_组内最大值/当月天数		数值型	手动衍生_py	python				called_days / days_month				
roam_out_days	国际漫游通话天数（含港澳台）_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
roam_out_days_p	国际漫游通话天数（含港澳台）_组内最大值/当月天数		数值型	手动衍生_py	python				roam_out_days / days_month				
roam_dmstc_days	省外漫游通话天数（国内且省外）_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
roam_dmstc_days_p	省外漫游通话天数（国内且省外）_组内最大值/当月天数		数值型	手动衍生_py	python				roam_dmstc_days / days_month				
local_days	市内通话天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
local_days_p	市内通话天数_组内最大值/当月天数		数值型	手动衍生_py	python				local_days / days_month				
ct_dur	网内通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
wj_dur	网间通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
ct_calling_dur	网内主叫时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
ct_cnt	网内通话次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
ct_calling_cnt	网内主叫次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
wj_calling_dur	网间主叫时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
wj_cnt	网间通话次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
wj_calling_cnt	网间主叫次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
jwq_num	通话交往圈用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
jwq_ct_num	通话交往圈网内用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
jwq_wj_num	通话交往圈网间用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
jwq_calling_num	主叫交往圈用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
jwq_ct_calling_num	主叫交往圈网内用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
jwq_wj_calling_num	主叫交往圈网间用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
jwq_called_num	被叫交往圈用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
jwq_ct_called_num	被叫交往圈网内用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
jwq_wj_called_num	被叫交往圈网间用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
data_flux	流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
last_to_acct_flux	上月递延本月流量_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
acct_to_next_flux	本月递延下月流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
flux_5g	5G流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
flux_5g_p	5G流量_组内总和/流量_组内总和		数值型	手动衍生_py	python				flux_5g / data_flux				
flux_4g	4G流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m						是		
flux_4g_p	4G流量_组内总和/流量_组内总和		数值型	手动衍生_py	python				flux_4g / data_flux				
flux_3g	3G流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
flux_3g_p	3G流量_组内总和/流量_组内总和		数值型	手动衍生_py	python				flux_3g / data_flux				
flux_2g	2G流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
flux_2g_p	2G流量_组内总和/流量_组内总和		数值型	手动衍生_py	python				flux_2g / data_flux				
int_flux	国际漫游流量（含港澳台）_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
dmstc_flux	省外漫游流量（国内且省外）_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
flux_days	流量使用天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
flux_days_p	流量使用天数_组内最大值/当月天数		数值型	手动衍生_py	python				flux_days / days_month				
int_flux_days	国际漫游流量使用天数（含港澳台）_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
int_flux_days_p	国际漫游流量使用天数（含港澳台）_组内最大值/流量_组内总和		数值型	手动衍生_py	python				int_flux_days / days_month				
dmstc_flux_days	省外漫游流量使用天数（国内且省外）_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m					不参与自动衍生		删除	否
dmstc_flux_days_p	省外漫游流量使用天数（国内且省外）_组内最大值/当月天数		数值型	手动衍生_py	python				dmstc_flux_days / days_month				
normal_flux	日间流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
normal_flux_days	日间流量使用天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
normal_flux_days_p	日间流量使用天数_组内最大值/当月天数		数值型	手动衍生_py	python				normal_flux_days / days_month				
night_flux	夜间流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
night_flux_days	夜间流量使用天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用				不参与自动衍生		删除	否
night_flux_days_p	夜间流量使用天数_组内最大值/当月天数		数值型	手动衍生_py	python				night_flux_days / days_month				
weekday_flux_avg	工作日日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
weekend_flux_avg	节假日日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
sx_flux_avg	上旬日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
zx_flux_avg	中旬日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
xx_flux_avg	下旬日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
mbl_out_flow	溢出流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
mbl_out_flow_p	流量费_组内总和 / 溢出流量_组内总和		数值型	手动衍生_py	python				flux_fee / mbl_out_flow				
sms_cnt	短信次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
sms_send_cnt	短信上行次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
sms_recv_cnt	短信下行次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
self_scs_cnt	本网客服--咨询次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m								
self_scs_tsbz_cnt	本网客服--投诉报障次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
call_turn_cnt	呼转次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_info_m	不可用							
paste_channel_prod_open	渠道、套餐、入网时长：交叉		类别型	手动衍生_py	python				(dev_channel_nbr, sig_offer_spec_name, open_months)				
valuecnt__is_acct~1	近n月出账月份数		数值型	自动衍生_py	python								
valuecnt__std_user_status_name~100000	近n月状态在用月份数		数值型	自动衍生_py	python								
valuecnt__is_valid~1	近n月is_valid=1月份数		数值型	自动衍生_py	python								
add_prodspecfluw_packflowsum	套内流量_组内主卡取值+订购包总流量_组内总和		数值型	手动衍生_py	python				prod_spec_fluw + pack_flow_sum				
add_prodspecfluw_packflowsum_lasttoacctflux	套内流量_组内主卡取值+订购包总流量_组内总和+上月递延本月流量_组内主卡取值		数值型	手动衍生_py	python				prod_spec_fluw + pack_flow_sum + last_to_acct_flux				
data_flux_p1	流量饱和度1		数值型	手动衍生_py	python				data_flux / prod_spec_fluw		是		
data_flux_p2	流量饱和度2		数值型	手动衍生_py	python				data_flux / add_prodspecfluw_packflowsum		是		
data_flux_p3	流量饱和度3		数值型	手动衍生_py	python				data_flux / add_prodspecfluw_packflowsum_lasttoacctflux		是		
casewhen_dur_flux_sms	通话时长_组内总和、流量_组内总和、短信次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{dur==0 & data_flux==0 & sms_cnt==0: 1,  else: 0}				
casewhen_calling_flux_sms	主叫时长_组内总和、流量_组内总和、短信次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{called_dur==0 & data_flux==0 & sms_cnt==0: 1,  else: 0}				
casewhen_called_flux_sms	被叫时长_组内总和、流量_组内总和、短信次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{calling_dur==0 & data_flux==0 & sms_cnt==0: 1,  else: 0}				
casewhen_dur_flux_send	通话时长_组内总和、流量_组内总和、短信上行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{dur==0 & data_flux==0 & sms_send_cnt==0: 1,  else: 0}				
casewhen_dur_flux_recv	通话时长_组内总和、流量_组内总和、短信下行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{dur==0 & data_flux==0 & sms_recv_cnt==0: 1,  else: 0}				
casewhen_calling_flux_send	主叫时长_组内总和、流量_组内总和、短信上行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{called_dur==0 & data_flux==0 & sms_send_cnt==0: 1,  else: 0}				
casewhen_calling_flux_recv	主叫时长_组内总和、流量_组内总和、短信下行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{called_dur==0 & data_flux==0 & sms_recv_cnt==0: 1,  else: 0}				
casewhen_called_flux_send	被叫时长_组内总和、流量_组内总和、短信上行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{calling_dur==0 & data_flux==0 & sms_send_cnt==0: 1,  else: 0}				
casewhen_called_flux_recv	被叫时长_组内总和、流量_组内总和、短信下行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{calling_dur==0 & data_flux==0 &  sms_recv_cnt==0: 1,  else: 0}				
casewhen_dur_flux	通话时长_组内总和、流量_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{dur==0 & data_flux==0: 1,  else: 0}				
casewhen_calling_flux	主叫时长_组内总和、流量_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{called_dur==0 & data_flux==0: 1,  else: 0}				
casewhen_called_flux	被叫时长_组内总和、流量_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{calling_dur==0 & data_flux==0: 1,  else: 0}				
spec_count0_fk_num_p	单产品套餐ID_组内主卡取值：分组统计副卡数大于0的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count0_fk_num / sig_offer_spec_id_count				
spec_count1_fk_num_p	单产品套餐ID_组内主卡取值：分组统计副卡数大于1的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count1_fk_num / sig_offer_spec_id_count				
spec_count2_fk_num_p	单产品套餐ID_组内主卡取值：分组统计副卡数大于2的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count2_fk_num / sig_offer_spec_id_count				
spec_pre025_fk_num_sub	单产品套餐ID_组内主卡取值：分组统计副卡数的25%分位数-副卡数量_组内主卡取值 		数值型	手动衍生_py	python			不可用	spec_pre025_fk_num - fk_num				
spec_pre05_fk_num_sub	单产品套餐ID_组内主卡取值：分组统计副卡数的50%分位数-副卡数量_组内主卡取值 		数值型	手动衍生_py	python			不可用	spec_pre05_fk_num - fk_num				
spec_max_fk_num_sub	单产品套餐ID_组内主卡取值：分组统计副卡数的众数-副卡数量_组内主卡取值 		数值型	手动衍生_py	python			不可用	spec_max_fk_num - fk_num				
spec_count0_comp_wf_num_p	单产品套餐ID_组内主卡取值：分组统计WiFi数大于0的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count0_comp_wf_num / sig_offer_spec_id_count				
spec_count1_comp_wf_num_p	单产品套餐ID_组内主卡取值：分组统计WiFi数大于1的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count1_comp_wf_num / sig_offer_spec_id_count				
spec_count2_comp_wf_num_p	单产品套餐ID_组内主卡取值：分组统计WiFi数大于2的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count2_comp_wf_num / sig_offer_spec_id_count				
spec_pre025_comp_wf_num_sub	单产品套餐ID_组内主卡取值：分组统计WiFi数的25%分位数-融合WiFi数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_pre025_comp_wf_num - comp_wf_num				
spec_pre05_comp_wf_num_sub	单产品套餐ID_组内主卡取值：分组统计WiFi数的50%分位数-融合WiFi数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_pre05_comp_wf_num - comp_wf_num				
spec_max_comp_wf_num_sub	单产品套餐ID_组内主卡取值：分组统计WiFi数的最大值-融合WiFi数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_max_comp_wf_num - comp_wf_num				
spec_count0_comp_kj_num_p	单产品套餐ID_组内主卡取值：分组统计看家数大于0的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count0_comp_kj_num / sig_offer_spec_id_count				
spec_count1_comp_kj_num_p	单产品套餐ID_组内主卡取值：分组统计看家数大于1的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count1_comp_kj_num / sig_offer_spec_id_count				
spec_count2_comp_kj_num_p	单产品套餐ID_组内主卡取值：分组统计看家数大于2的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count2_comp_kj_num / sig_offer_spec_id_count				
spec_pre025_comp_kj_num_sub	单产品套餐ID_组内主卡取值：分组统计看家数的50%分位数-融合看家数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_pre025_comp_kj_num - comp_kj_num				
spec_pre05_comp_kj_num_sub	单产品套餐ID_组内主卡取值：分组统计看家数的25%分位数-融合看家数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_pre05_comp_kj_num - comp_kj_num				
spec_max_comp_kj_num_sub	单产品套餐ID_组内主卡取值：分组统计看家数的最大值-融合看家数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_max_comp_kj_num - comp_kj_num				
rn	随机排序字段	bigint	数值型	手动衍生_sql	sql					不参与自动衍生		删除	
acct_month	账期分区键（分区键不是表中字段的情况）	varchar(6)	类别型	其他									
"""
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# ------------------------------- <editor-fold desc="移网field_base_套餐组"> -------------------------------------------
s_yw_team = """field_name	comment	dtype_db	dtype_classify	field_src	table	available	available_notzd	available_zd	formula	remark	must_remain	into_model	is_cause
month_id	账期	varchar(6)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m							删除	否
user_no_teamzk	用户ID_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
acct_id_teamzk	账户ID_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
customer_no_teamzk	客户ID_组内主卡取值	varchar(30)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
dev_channel_nbr_teamzk	入网渠道ID_组内主卡取值	varchar(30)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	否
dev_channel_name_teamzk	入网渠道名称_组内主卡取值	varchar(500)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生		删除	否
open_date_teamzk	入网日期_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
open_months_teamzk	入网时长(月)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生	是		
cert_type_teamzk	证件类型(1:个人,0:非个人)_组内主卡取值	varchar(1)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
cert_nbr_teamzk	身份证号码_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	否
area_no_teamzk	地市名称_组内主卡取值	varchar(5)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			否
city_no_teamzk	区县名称_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			否
is_village_teamzk	农村标识_组内主卡取值	varchar(1)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
age_teamzk	年龄_组内主卡取值	varchar(3)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生			
sex_teamzk	性别_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生			否
is_valid_teamzk	是否在网_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			否
std_user_status_name_teamzk	用户状态名称_组内主卡取值	varchar(250)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
stop_times_teamadd	停机次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
resent_stop_date_teammax	最近一次停机日期_组内最大值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
monthsaready_resent_stop_date_teammax	最近一次停机日期_组内最大值_已发生时长		数值型	手动衍生_py	python				current_date - resent_stop_date_teammax	不参与近n月自动衍生			
is_acct_teamzk	是否出账_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
payment_mode_cd_teamzk	付费方式_组内主卡取值	varchar(10)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
is_real_name_teamzk	是否实名用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生			
is_gz_teamzk	是否公众用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_xy_teamzk	是否校园用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_gov_teamzk	是否政企用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_grp_member_teamzk	是否集团成员_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_hlwk_teamzk	是否互联网卡_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_sw_teamzk	是否三无用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_red_list_teamzk	剔除口径类字段_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_5g_main_offer_teamzk	是否5G主销售品用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
is_5g_upgrade_offer_teamzk	是否5G流量包用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
is_fq_teamzk	是否分期用户_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
is_usim_teamzk	是否usim卡_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
tyaddresscode_teamzk	小区编码_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	
tyaddressname_teamzk	小区名称_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	
shentoulv_teamzk	小区渗透率_组内主卡取值	varchar(10)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
family_form_ersj_teamzk	家庭构成_二人世界_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生	是		
family_form_skzj_teamzk	家庭构成_三口之家_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生	是		
family_form_unknown_teamzk	家庭构成_未知_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生	是		
family_prop_child_teamzk	家庭特征_家有儿童_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生	是		
family_prop_elder_teamzk	家庭特征_家有老人_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生	是		
family_prop_unknown_teamzk	家庭特征_未知_组内主卡取值	varchar(2)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生	是		
family_type_teamzk	家庭类型_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生	是		
sig_offer_spec_id_teamzk	单产品套餐ID_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	否
sig_offer_spec_name_teamzk	单产品套餐名称_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	
sig_offer_create_dt_teamzk	单产品套餐订购时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
monthsaready_sig_offer_create_dt_teamzk	单产品套餐订购时间_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - sig_offer_create_dt_teamzk	不参与近n月自动衍生			
comp_offer_spec_id_teamzk	融合套餐ID_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	否
comp_offer_spec_name_teamzk	融合套餐名称_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	
comp_offer_create_dt_teamzk	融合套餐订购时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
monthsaready_comp_offer_create_dt_teamzk	融合套餐订购时间_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - comp_offer_create_dt_teamzk	不参与近n月自动衍生			
is_dupety_teamzk	主副卡类型(0:主卡,1:副卡)_组内主卡取值	varchar(1)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
fk_num_teamzk	副卡数量_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
zk_user_no	主卡ID	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生			否
prod_spec_fee_teamzk	套餐月费_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
prod_spec_fluw_teamzk	套内流量_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
prod_spec_dur_teamzk	套内语音_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
mbl_inner_fluw_teamadd	套内流量使用量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
dialing_inner_teamadd	套内语音使用量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
is_comp_teamzk	是否融合业务_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
comp_user_no_teamzk	融合组ID_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
comp_eff_date_teamzk	融合业务生效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
monthsaready_comp_eff_date_teamzk	融合业务生效时间_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - comp_eff_date_teamzk	不参与近n月自动衍生			
comp_exp_date_teamzk	融合业务失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
monthsremain_comp_exp_date_teamzk	融合业务失效时间_组内主卡取值_剩余时长		数值型	手动衍生_py	python				comp_exp_date_teamzk - current_date	不参与近n月自动衍生			
comp_type_teamzk	融合产品结构_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
comp_num_teamzk	融合产品数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
comp_kd_num_teamzk	融合宽带数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
comp_yd_num_teamzk	融合移动数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
comp_hd_num_teamzk	融合电视数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
comp_wf_num_teamzk	融合WiFi数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
comp_kj_num_teamzk	融合看家数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
is_agre_teamzk	是否有合约_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
track_eff_date_teamzk	合约生效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
monthsaready_track_eff_date_teamzk	合约生效时间_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - track_eff_date_teamzk	不参与近n月自动衍生			
track_exp_date_teamzk	合约失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
monthsremain_track_exp_date_teamzk	合约失效时间_组内主卡取值_剩余时长		数值型	手动衍生_py	python				track_exp_date_teamzk - current_date	不参与近n月自动衍生			
pack_hd_num_teamadd	订购电视包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
pack_dur_num_teamadd	订购语音包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
pack_flow_num_teamadd	订购流量包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
pack_flow_sum_teamadd	订购包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
pack_dur_sum_teamadd	订购包总语音_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
pack_month_flow_num_teamadd	订购流量月包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
pack_month_flow_sum_teamadd	订购流量月包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
pack_month_flow_num_m_teamadd	当月订购流量月包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用					是		
pack_month_flow_sum_m_teamadd	当月订购流量月包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用					是		
pack_month_flow_exp_date_teamzk	办理流量月包失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
pack_directed_num_teamadd	订购定向流量包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
pack_directed_sum_teamadd	订购定向流量包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
pack_directed_num_m_teamadd	当月订购定向流量包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
pack_directed_sum_m_teamadd	当月订购定向流量包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
pack_directed_exp_date_teamzk	办理定向流量包失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
pack_5g_internet_num_teamadd	订购5G网络包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
pack_5g_internet_free_num_teamadd	订购0元5G网络包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
pack_5g_internet_sum_teamadd	订购5G网络包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
pack_5g_internet_num_m_teamadd	当月订购5G网络包总数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
pack_5g_internet_free_num_m_teamadd	当月订购0元5G网络包总数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生			
pack_5g_internet_sum_m_teamadd	当月订购5G网络包总流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
pack_5g_internet_exp_date_teamzk	办理5G包失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
pack_spcl_num_teamadd	订购视频彩铃包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
pack_spcl_free_num_teamadd	订购0元视频彩铃包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
pack_spcl_num_m_teamadd	当月订购视频彩铃包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
pack_spcl_free_num_m_teamadd	当月订购0元视频彩铃包数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生			
pack_spcl_exp_date_teamzk	办理视频彩铃失效时间_组内主卡取值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
monthsremain_pack_spcl_exp_date	办理视频彩铃失效时间_组内主卡取值_剩余时长		数值型	手动衍生_py	python				pack_spcl_exp_date_teamzk - current_date	不参与近n月自动衍生			
pack_spcl_6_month_teammax	近半年是否办理过视频彩铃_组内最大值	varchar(2)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生			
is_hy_or_hb_teamzk	是否订购合约及红包类销售品_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
point_teamzk	积分_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
star_teamzk	星级_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
basic_credit_teamzk	信用度_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
quota_teamzk	授信额度_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
is_lh_teamzk	是否靓号_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
lh_type_teamzk	靓号类型_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
is_bxl_teamzk	当前是否不限量_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用					是		
bxl_flow_step_teamzk	不限量流量档位_组内主卡取值	varchar(20)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
bxl_deal_times_teamzk	不限量包的办理次数_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
is_deal_bxl_teamzk	是否曾办理过不限量包_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
pack_bxl_num_teamzk	当前不限量包数量_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
owe_flag_teamzk	当前是否欠费_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
owe_charge_teamadd	当前欠费金额_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
owe_times_teamadd	欠费次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
last_owe_month_teammax	最近一次欠费时间_组内最大值	varchar(6)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
monthsaready_last_owe_month_teammax	最近一次欠费时间_组内最大值_已发生时长		数值型	手动衍生_py	python				current_date - last_owe_month_teammax	不参与近n月自动衍生			
last_owe_charge_teamzk	最近一次欠费金额_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生			
owe_charge_acct_teamadd	当月产生欠费总金额_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
payment_cnt_teamadd	缴费次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
payment_charge_teamadd	缴费金额_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
payment_time_teammax	最近一次缴费时间_组内最大值	varchar(14)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
monthsaready_payment_time_teammax	最近一次缴费时间_组内最大值_已发生时长		数值型	手动衍生_py	python				current_date - payment_time_teammax	不参与近n月自动衍生			
payment_fee_teamzk	最近一次缴费金额_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与近n月自动衍生			
balance_teamadd	余额_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用					是		
arpu_teamadd	ARPU_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
arpu_teamadd_p	arpu/余额		数值型	手动衍生_py	python				arpu_teamadd / balance_teamadd		是		
call_fee_teamadd	通话费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
call_local_fee_teamadd	本地通话费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
call_long_prov_fee_teamadd	长途通话费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
call_roam_fee_teamadd	漫游通话费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
flux_fee_teamadd	流量费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用					是		
sms_fee_teamadd	短信费_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
rgst_tmn_no_teamzk	终端IMEI_组内主卡取值	varchar(10)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	
rgst_tmn_brand_teamzk	终端品牌_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
rgst_tmn_model_teamzk	终端型号_组内主卡取值	varchar(32)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生		删除	
rgst_tmn_type_teamzk	终端类型_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
rgst_tmn_time_teamzk	注册日期_组内主卡取值	varchar(20)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
monthsaready_rgst_tmn_time_teamzk	注册日期_组内主卡取值_已发生时长		数值型	手动衍生_py	python				current_date - rgst_tmn_time_teamzk	不参与近n月自动衍生			
rgst_tmn_flag_teamzk	是否智能机(1:是 0:否)_组内主卡取值	varchar(20)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
phone_price_teamzk	价格_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
operating_sys_teamzk	操作系统_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
market_time_teamzk	上市时间_组内主卡取值	varchar(255)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
prd_position_teamzk	产品定位_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
screens_nbr_teamzk	屏幕数量_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
main_screen_size_teamzk	主屏幕尺寸_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
baseband_chip_clocked_teamzk	基带芯片主频_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
resolution_teamzk	显示分辩率_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
ram_teamzk	RAM_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用	不可用			不参与近n月自动衍生			
main_camera_teamzk	主摄像头_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用	不可用			不参与近n月自动衍生			
start_use_time_teamzk	开始使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
end_use_time_teamzk	结束使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
usage_days_teamzk	当次使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
usage_days_sum_teamzk	总计使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_in_use_teamzk	当前是否正在使用_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_new_tmn_teamzk	是否为新终端（按品牌）_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
rgst_tmn_brand_1_teamzk	上一个终端_终端品牌_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_model_1_teamzk	上一个终端_终端型号_组内主卡取值	varchar(32)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生		删除	
rgst_tmn_type_1_teamzk	上一个终端_终端类型_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_time_1_teamzk	上一个终端_注册日期_组内主卡取值	varchar(20)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
rgst_tmn_flag_1_teamzk	上一个终端_是否智能机(1:是 0:否)_组内主卡取值	varchar(20)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
phone_price_1_teamzk	上一个终端_价格_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
operating_sys_1_teamzk	上一个终端_操作系统_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
market_time_1_teamzk	上一个终端_上市时间_组内主卡取值	varchar(255)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
prd_position_1_teamzk	上一个终端_产品定位_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
screens_nbr_1_teamzk	上一个终端_屏幕数量_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
main_screen_size_1_teamzk	上一个终端_主屏幕尺寸_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
baseband_chip_clocked_1_teamzk	上一个终端_基带芯片主频_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
resolution_1_teamzk	上一个终端_显示分辩率_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
ram_1_teamzk	上一个终端_RAM_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用	不可用			不参与近n月自动衍生			
main_camera_1_teamzk	上一个终端_主摄像头_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用	不可用			不参与近n月自动衍生			
start_use_time_1_teamzk	上一个终端_开始使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
end_use_time_1_teamzk	上一个终端_结束使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
usage_days_1_teamzk	上一个终端_当次使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
usage_days_sum_1_teamzk	上一个终端_总计使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_new_tmn_1_teamzk	上一个终端_是否为新终端（按品牌）_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_brand_2_teamzk	上两个终端_终端品牌_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_model_2_teamzk	上两个终端_终端型号_组内主卡取值	varchar(32)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生		删除	
rgst_tmn_type_2_teamzk	上两个终端_终端类型_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
rgst_tmn_time_2_teamzk	上两个终端_注册日期_组内主卡取值	varchar(20)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
rgst_tmn_flag_2_teamzk	上两个终端_是否智能机(1:是 0:否)_组内主卡取值	varchar(20)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
phone_price_2_teamzk	上两个终端_价格_组内主卡取值	varchar(50)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
operating_sys_2_teamzk	上两个终端_操作系统_组内主卡取值	varchar(255)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
market_time_2_teamzk	上两个终端_上市时间_组内主卡取值	varchar(255)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
prd_position_2_teamzk	上两个终端_产品定位_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
screens_nbr_2_teamzk	上两个终端_屏幕数量_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
main_screen_size_2_teamzk	上两个终端_主屏幕尺寸_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
baseband_chip_clocked_2_teamzk	上两个终端_基带芯片主频_组内主卡取值	varchar(100)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
resolution_2_teamzk	上两个终端_显示分辩率_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
ram_2_teamzk	上两个终端_RAM_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用	不可用			不参与近n月自动衍生			
main_camera_2_teamzk	上两个终端_主摄像头_组内主卡取值	varchar(200)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用	不可用			不参与近n月自动衍生			
start_use_time_2_teamzk	上两个终端_开始使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
end_use_time_2_teamzk	上两个终端_结束使用时间_组内主卡取值	varchar(8)	日期型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与自动衍生		删除	否
usage_days_2_teamzk	上两个终端_当次使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
usage_days_sum_2_teamzk	上两个终端_总计使用时长(天)_组内主卡取值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
is_new_tmn_2_teamzk	上两个终端_是否为新终端（按品牌）_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m		不可用			不参与近n月自动衍生			
is_shaungka_teamzk	是否双卡终端_组内主卡取值	varchar(1)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
shaungka_type_teamzk	双卡终端的卡类型_组内主卡取值	varchar(10)	类别型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与近n月自动衍生			
dur_teamadd	通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
calling_dur_teamadd	主叫时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
called_dur_teamadd	被叫时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
cnt_teamadd	通话次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
calling_cnt_teamadd	主叫次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
called_cnt_teamadd	被叫次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
roam_dur_teamadd	漫游通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
roam_cnt_teamadd	漫游通话次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
roam_out_dur_teamadd	国际漫游通话时长（含港澳台）_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
roam_dmstc_dur_teamadd	省外漫游通话时长（国内且省外）_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
local_dur_teamadd	市内通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
call_days_teammax	通话天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
call_days_teammax_p	通话天数_组内最大值/当月天数		数值型	手动衍生_py	python				call_days_teammax / days_month				
calling_days_teammax	主叫天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
calling_days_teammax_p	主叫天数_组内最大值/当月天数		数值型	手动衍生_py	python				calling_days_teammax / days_month				
called_days_teammax	被叫天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
called_days_teammax_p	被叫天数_组内最大值/当月天数		数值型	手动衍生_py	python				called_days_teammax / days_month				
roam_out_days_teammax	国际漫游通话天数（含港澳台）_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
roam_out_days_teammax_p	国际漫游通话天数（含港澳台）_组内最大值/当月天数		数值型	手动衍生_py	python				roam_out_days_teammax / days_month				
roam_dmstc_days_teammax	省外漫游通话天数（国内且省外）_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
roam_dmstc_days_teammax_p	省外漫游通话天数（国内且省外）_组内最大值/当月天数		数值型	手动衍生_py	python				roam_dmstc_days_teammax / days_month				
local_days_teammax	市内通话天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
local_days_teammax_p	市内通话天数_组内最大值/当月天数		数值型	手动衍生_py	python				local_days_teammax / days_month				
ct_dur_teamadd	网内通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
wj_dur_teamadd	网间通话时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
ct_calling_dur_teamadd	网内主叫时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
ct_cnt_teamadd	网内通话次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
ct_calling_cnt_teamadd	网内主叫次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
wj_calling_dur_teamadd	网间主叫时长_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
wj_cnt_teamadd	网间通话次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
wj_calling_cnt_teamadd	网间主叫次数_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
jwq_num_teamadd	通话交往圈用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
jwq_ct_num_teamadd	通话交往圈网内用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
jwq_wj_num_teamadd	通话交往圈网间用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
jwq_calling_num_teamadd	主叫交往圈用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
jwq_ct_calling_num_teamadd	主叫交往圈网内用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
jwq_wj_calling_num_teamadd	主叫交往圈网间用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
jwq_called_num_teamadd	被叫交往圈用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
jwq_ct_called_num_teamadd	被叫交往圈网内用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
jwq_wj_called_num_teamadd	被叫交往圈网间用户数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
data_flux_teamadd	流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
last_to_acct_flux_teamzk	上月递延本月流量_组内主卡取值	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
acct_to_next_flux_teamadd	本月递延下月流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
flux_5g_teamadd	5G流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
flux_5g_teamadd_p	5G流量_组内总和/流量_组内总和		数值型	手动衍生_py	python				flux_5g_teamadd / data_flux_teamadd				
flux_4g_teamadd	4G流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m						是		
flux_4g_teamadd_p	4G流量_组内总和/流量_组内总和		数值型	手动衍生_py	python				flux_4g_teamadd / data_flux_teamadd				
flux_3g_teamadd	3G流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
flux_3g_teamadd_p	3G流量_组内总和/流量_组内总和		数值型	手动衍生_py	python				flux_3g_teamadd / data_flux_teamadd				
flux_2g_teamadd	2G流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
flux_2g_teamadd_p	2G流量_组内总和/流量_组内总和		数值型	手动衍生_py	python				flux_2g_teamadd / data_flux_teamadd				
int_flux_teamadd	国际漫游流量（含港澳台）_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
dmstc_flux_teamadd	省外漫游流量（国内且省外）_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
flux_days_teammax	流量使用天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
flux_days_teammax_p	流量使用天数_组内最大值/当月天数		数值型	手动衍生_py	python				flux_days_teammax / days_month				
int_flux_days_teammax	国际漫游流量使用天数（含港澳台）_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
int_flux_days_teammax_p	国际漫游流量使用天数（含港澳台）_组内最大值/流量_组内总和		数值型	手动衍生_py	python				int_flux_days_teammax / days_month				
dmstc_flux_days_teammax	省外漫游流量使用天数（国内且省外）_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m					不参与自动衍生		删除	否
dmstc_flux_days_teammax_p	省外漫游流量使用天数（国内且省外）_组内最大值/当月天数		数值型	手动衍生_py	python				dmstc_flux_days_teammax / days_month				
normal_flux_teamadd	日间流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
normal_flux_days_teammax	日间流量使用天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
normal_flux_days_teammax_p	日间流量使用天数_组内最大值/当月天数		数值型	手动衍生_py	python				normal_flux_days_teammax / days_month				
night_flux_teamadd	夜间流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
night_flux_days_teammax	夜间流量使用天数_组内最大值	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用				不参与自动衍生		删除	否
night_flux_days_teammax_p	夜间流量使用天数_组内最大值/当月天数		数值型	手动衍生_py	python				night_flux_days_teammax / days_month				
weekday_flux_avg_teamadd	工作日日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
weekend_flux_avg_teamadd	节假日日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
sx_flux_avg_teamadd	上旬日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
zx_flux_avg_teamadd	中旬日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
xx_flux_avg_teamadd	下旬日均流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
mbl_out_flow_teamadd	溢出流量_组内总和	decimal(20,4)	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
mbl_out_flow_teamadd_p	流量费_组内总和 / 溢出流量_组内总和		数值型	手动衍生_py	python				flux_fee_teamadd / mbl_out_flow_teamadd				
sms_cnt_teamadd	短信次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
sms_send_cnt_teamadd	短信上行次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
sms_recv_cnt_teamadd	短信下行次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
self_scs_cnt_teamadd	本网客服--咨询次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m								
self_scs_tsbz_cnt_teamadd	本网客服--投诉报障次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
call_turn_cnt_teamadd	呼转次数_组内总和	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_info_m	不可用							
paste_channel_prod_open_teamzk	渠道、套餐、入网时长：交叉		类别型	手动衍生_py	python				(dev_channel_nbr_teamzk, sig_offer_spec_name_teamzk, open_months_teamzk)				
valuecnt__is_acct_teamzk~1	近n月出账月份数		数值型	自动衍生_py	python								
valuecnt__std_user_status_name_teamzk~100000	近n月状态在用月份数		数值型	自动衍生_py	python								
valuecnt__is_valid_teamzk~1	近n月is_valid=1月份数		数值型	自动衍生_py	python								
add_prodspecfluw_packflowsum	套内流量_组内主卡取值+订购包总流量_组内总和		数值型	手动衍生_py	python				prod_spec_fluw_teamzk + pack_flow_sum_teamadd				
add_prodspecfluw_packflowsum_lasttoacctflux	套内流量_组内主卡取值+订购包总流量_组内总和+上月递延本月流量_组内主卡取值		数值型	手动衍生_py	python				prod_spec_fluw_teamzk + pack_flow_sum_teamadd + last_to_acct_flux_teamzk				
data_flux_teamadd_p1	流量饱和度1		数值型	手动衍生_py	python				data_flux_teamadd / prod_spec_fluw_teamzk		是		
data_flux_teamadd_p2	流量饱和度2		数值型	手动衍生_py	python				data_flux_teamadd / add_prodspecfluw_packflowsum		是		
data_flux_teamadd_p3	流量饱和度3		数值型	手动衍生_py	python				data_flux_teamadd / add_prodspecfluw_packflowsum_lasttoacctflux		是		
casewhen_dur_flux_sms	通话时长_组内总和、流量_组内总和、短信次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{dur_teamadd==0 & data_flux_teamadd==0 & sms_cnt_teamadd==0: 1,  else: 0}				
casewhen_calling_flux_sms	主叫时长_组内总和、流量_组内总和、短信次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{called_dur_teamadd==0 & data_flux_teamadd==0 & sms_cnt_teamadd==0: 1,  else: 0}				
casewhen_called_flux_sms	被叫时长_组内总和、流量_组内总和、短信次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{calling_dur_teamadd==0 & data_flux_teamadd==0 & sms_cnt_teamadd==0: 1,  else: 0}				
casewhen_dur_flux_send	通话时长_组内总和、流量_组内总和、短信上行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{dur_teamadd==0 & data_flux_teamadd==0 & sms_send_cnt_teamadd==0: 1,  else: 0}				
casewhen_dur_flux_recv	通话时长_组内总和、流量_组内总和、短信下行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{dur_teamadd==0 & data_flux_teamadd==0 & sms_recv_cnt_teamadd==0: 1,  else: 0}				
casewhen_calling_flux_send	主叫时长_组内总和、流量_组内总和、短信上行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{called_dur_teamadd==0 & data_flux_teamadd==0 & sms_send_cnt_teamadd==0: 1,  else: 0}				
casewhen_calling_flux_recv	主叫时长_组内总和、流量_组内总和、短信下行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{called_dur_teamadd==0 & data_flux_teamadd==0 & sms_recv_cnt_teamadd==0: 1,  else: 0}				
casewhen_called_flux_send	被叫时长_组内总和、流量_组内总和、短信上行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{calling_dur_teamadd==0 & data_flux_teamadd==0 & sms_send_cnt_teamadd==0: 1,  else: 0}				
casewhen_called_flux_recv	被叫时长_组内总和、流量_组内总和、短信下行次数_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{calling_dur_teamadd==0 & data_flux_teamadd==0 &  sms_recv_cnt_teamadd==0: 1,  else: 0}				
casewhen_dur_flux	通话时长_组内总和、流量_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{dur_teamadd==0 & data_flux_teamadd==0: 1,  else: 0}				
casewhen_calling_flux	主叫时长_组内总和、流量_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{called_dur_teamadd==0 & data_flux_teamadd==0: 1,  else: 0}				
casewhen_called_flux	被叫时长_组内总和、流量_组内总和：均为0，则打标为1		数值型	手动衍生_py	python				{calling_dur_teamadd==0 & data_flux_teamadd==0: 1,  else: 0}				
sig_offer_spec_id_teamzk_count	单产品套餐ID_组内主卡取值：分组计数	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count0_fk_num_teamzk	单产品套餐ID_组内主卡取值：分组统计副卡数大于0的量	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count1_fk_num_teamzk	单产品套餐ID_组内主卡取值：分组统计副卡数大于1的量	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count2_fk_num_teamzk	单产品套餐ID_组内主卡取值：分组统计副卡数大于2的量	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_pre025_fk_num_teamzk	单产品套餐ID_组内主卡取值：分组统计副卡数的25%分位数	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_pre05_fk_num_teamzk	单产品套餐ID_组内主卡取值：分组统计副卡数的50%分位数	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_max_fk_num_teamzk	单产品套餐ID_组内主卡取值：分组统计副卡数的众数	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count0_comp_wf_num_teamzk	单产品套餐ID_组内主卡取值：分组统计WiFi数大于0的量	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count1_comp_wf_num_teamzk	单产品套餐ID_组内主卡取值：分组统计WiFi数大于1的量	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count2_comp_wf_num_teamzk	单产品套餐ID_组内主卡取值：分组统计WiFi数大于2的量	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_pre025_comp_wf_num_teamzk	单产品套餐ID_组内主卡取值：分组统计WiFi数的25%分位数	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_pre05_comp_wf_num_teamzk	单产品套餐ID_组内主卡取值：分组统计WiFi数的50%分位数	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_max_comp_wf_num_teamzk	单产品套餐ID_组内主卡取值：分组统计WiFi数的最大值	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count0_comp_kj_num_teamzk	单产品套餐ID_组内主卡取值：分组统计看家数大于0的量	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count1_comp_kj_num_teamzk	单产品套餐ID_组内主卡取值：分组统计看家数大于1的量	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count2_comp_kj_num_teamzk	单产品套餐ID_组内主卡取值：分组统计看家数大于2的量	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_pre025_comp_kj_num_teamzk	单产品套餐ID_组内主卡取值：分组统计看家数的50%分位数	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_pre05_comp_kj_num_teamzk	单产品套餐ID_组内主卡取值：分组统计看家数的25%分位数	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_max_comp_kj_num_teamzk	单产品套餐ID_组内主卡取值：分组统计看家数的最大值	decimal(20,4)	数值型	手动衍生_sql	kehujingyingbudb.dm_zc_yw_moxing_team_add_m			不可用					
spec_count0_fk_num_teamzk_p	单产品套餐ID_组内主卡取值：分组统计副卡数大于0的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count0_fk_num_teamzk / sig_offer_spec_id_teamzk_count				
spec_count1_fk_num_teamzk_p	单产品套餐ID_组内主卡取值：分组统计副卡数大于1的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count1_fk_num_teamzk / sig_offer_spec_id_teamzk_count				
spec_count2_fk_num_teamzk_p	单产品套餐ID_组内主卡取值：分组统计副卡数大于2的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count2_fk_num_teamzk / sig_offer_spec_id_teamzk_count				
spec_pre025_fk_num_teamzk_sub	单产品套餐ID_组内主卡取值：分组统计副卡数的25%分位数-副卡数量_组内主卡取值 		数值型	手动衍生_py	python			不可用	spec_pre025_fk_num_teamzk - fk_num_teamzk				
spec_pre05_fk_num_teamzk_sub	单产品套餐ID_组内主卡取值：分组统计副卡数的50%分位数-副卡数量_组内主卡取值 		数值型	手动衍生_py	python			不可用	spec_pre05_fk_num_teamzk - fk_num_teamzk				
spec_max_fk_num_teamzk_sub	单产品套餐ID_组内主卡取值：分组统计副卡数的众数-副卡数量_组内主卡取值 		数值型	手动衍生_py	python			不可用	spec_max_fk_num_teamzk - fk_num_teamzk				
spec_count0_comp_wf_num_teamzk_p	单产品套餐ID_组内主卡取值：分组统计WiFi数大于0的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count0_comp_wf_num_teamzk / sig_offer_spec_id_teamzk_count				
spec_count1_comp_wf_num_teamzk_p	单产品套餐ID_组内主卡取值：分组统计WiFi数大于1的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count1_comp_wf_num_teamzk / sig_offer_spec_id_teamzk_count				
spec_count2_comp_wf_num_teamzk_p	单产品套餐ID_组内主卡取值：分组统计WiFi数大于2的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count2_comp_wf_num_teamzk / sig_offer_spec_id_teamzk_count				
spec_pre025_comp_wf_num_teamzk_sub	单产品套餐ID_组内主卡取值：分组统计WiFi数的25%分位数-融合WiFi数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_pre025_comp_wf_num_teamzk - comp_wf_num_teamzk				
spec_pre05_comp_wf_num_teamzk_sub	单产品套餐ID_组内主卡取值：分组统计WiFi数的50%分位数-融合WiFi数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_pre05_comp_wf_num_teamzk - comp_wf_num_teamzk				
spec_max_comp_wf_num_teamzk_sub	单产品套餐ID_组内主卡取值：分组统计WiFi数的最大值-融合WiFi数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_max_comp_wf_num_teamzk - comp_wf_num_teamzk				
spec_count0_comp_kj_num_teamzk_p	单产品套餐ID_组内主卡取值：分组统计看家数大于0的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count0_comp_kj_num_teamzk / sig_offer_spec_id_teamzk_count				
spec_count1_comp_kj_num_teamzk_p	单产品套餐ID_组内主卡取值：分组统计看家数大于1的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count1_comp_kj_num_teamzk / sig_offer_spec_id_teamzk_count				
spec_count2_comp_kj_num_teamzk_p	单产品套餐ID_组内主卡取值：分组统计看家数大于2的量/分组计数		数值型	手动衍生_py	python			不可用	spec_count2_comp_kj_num_teamzk / sig_offer_spec_id_teamzk_count				
spec_pre025_comp_kj_num_teamzk_sub	单产品套餐ID_组内主卡取值：分组统计看家数的50%分位数-融合看家数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_pre025_comp_kj_num_teamzk - comp_kj_num_teamzk				
spec_pre05_comp_kj_num_teamzk_sub	单产品套餐ID_组内主卡取值：分组统计看家数的25%分位数-融合看家数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_pre05_comp_kj_num_teamzk - comp_kj_num_teamzk				
spec_max_comp_kj_num_teamzk_sub	单产品套餐ID_组内主卡取值：分组统计看家数的最大值-融合看家数_组内主卡取值		数值型	手动衍生_py	python			不可用	spec_max_comp_kj_num_teamzk - comp_kj_num_teamzk				
flag_ls	目标字段<流失预警模型_移网>	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_target_m								
flag_rh	目标字段<转融合模型_单C>	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_target_m								
flag_fk	目标字段<副卡加装模型_移网>	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_target_m								
flag_gq	目标字段<套餐高迁模型>	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_target_m								
flag_yb	目标字段<流量月包加装模型>	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_target_m								
flag_hj	目标字段<终端换机模型>	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_target_m								
flag_dxb	目标字段<加定向流量包模型>	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_target_m								
flag_5gb	目标字段<5G加包模型>	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_target_m								
flag_cl	目标字段<视频彩铃潜客模型>	int	数值型	原始	kehujingyingbudb.dm_zc_yw_moxing_team_target_m								
score_flag_ls	分数<流失预警模型_移网>	decimal(30,4)	数值型	原始	kehujingyingbudb.binaryclassify_score_m							删除	
ago_score_flag_ls	历史分数<流失预警模型_移网>	decimal(30,4)	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
score_flag_rh	分数<转融合模型_单C>	decimal(30,4)	数值型	原始	kehujingyingbudb.binaryclassify_score_m							删除	
ago_score_flag_rh	历史分数<转融合模型_单C>	decimal(30,4)	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
score_flag_fk	分数<副卡加装模型_移网>	decimal(30,4)	数值型	原始	kehujingyingbudb.binaryclassify_score_m							删除	
ago_score_flag_fk	历史分数<副卡加装模型_移网>	decimal(30,4)	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
score_flag_gq	分数<套餐高迁模型>	decimal(30,4)	数值型	原始	kehujingyingbudb.binaryclassify_score_m							删除	
ago_score_flag_gq	历史分数<套餐高迁模型>	decimal(30,4)	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
score_flag_yb	分数<流量月包加装模型>	decimal(30,4)	数值型	原始	kehujingyingbudb.binaryclassify_score_m							删除	
ago_score_flag_yb	历史分数<流量月包加装模型>	decimal(30,4)	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
score_flag_hj	分数<终端换机模型>	decimal(30,4)	数值型	原始	kehujingyingbudb.binaryclassify_score_m							删除	
ago_score_flag_hj	历史分数<终端换机模型>	decimal(30,4)	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
score_flag_dxb	分数<加定向流量包模型>	decimal(30,4)	数值型	原始	kehujingyingbudb.binaryclassify_score_m							删除	
ago_score_flag_dxb	历史分数<加定向流量包模型>	decimal(30,4)	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
score_flag_5gb	分数<5G加包模型>	decimal(30,4)	数值型	原始	kehujingyingbudb.binaryclassify_score_m							删除	
ago_score_flag_5gb	历史分数<5G加包模型>	decimal(30,4)	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
score_flag_cl	分数<视频彩铃潜客模型>	decimal(30,4)	数值型	原始	kehujingyingbudb.binaryclassify_score_m							删除	
ago_score_flag_cl	历史分数<视频彩铃潜客模型>	decimal(30,4)	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
rn	随机排序字段	bigint	数值型	手动衍生_sql	sql					不参与自动衍生		删除	
acct_month	账期分区键（分区键不是表中字段的情况）	varchar(6)	类别型	其他									
dayvalue_std_user_status	次月dd日用户状态	varchar(250)	类别型	原始	edww.dww_d_pr_pri_al_inst								
dayvalue_is_agre	次月dd日是否有合约	varchar(1)	数值型	原始	edww.dww_d_pr_pri_al_inst								
dayvalue_is_comp	次月dd日是否融合业务	varchar(1)	数值型	原始	edww.dww_d_pr_pri_al_inst								
day_id	日表账期字段	text	类别型	其他	edww.dww_d_pr_pri_al_inst															
"""
# </editor-fold>


#
#
# <editor-fold desc="宽带field_base_用户">

s_kd_user = """"""
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# ------------------------------------- <editor-fold desc="融合field_base_套餐组"> -------------------------------------
s_rh_team = """"""
# </editor-fold> -------------------------------------------------------------------------------------------------------


#
#
# -------------------------------------- <editor-fold desc="模型示例field_base"> ---------------------------------------
s_yw_fake = """field_name	comment	dtype_db	dtype_classify	field_src	table	available	available_notzd	available_zd	formula	remark	must_remain	into_model	is_cause
acct_month	账期	bigint	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								否
phone_no_null	手机号码	numeric	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m					不参与自动衍生		删除	否
phone_no_tm	模糊手机号	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m					不参与自动衍生		删除	否
user_id	手机号ID	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								否
dinner_id	主套餐策划ID	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m					不参与当月自动衍生			否
account_id	帐户编号	numeric	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m					不参与当月自动衍生	是		否
innet_date	入网日期	text	日期型	原始	kehujingyingbudb.ml_feature_info_yw_user_m							删除	否
innet_months	入网时长	bigint	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m					不参与近n月自动衍生			
age	年龄	bigint	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
sex	性别	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								否
age_add	table_a_add测试字段1	integer	数值型	手动衍生_sql	kehujingyingbudb.ml_feature_add_yw_user_m								
sex_add	table_a_add测试字段2	text	类别型	手动衍生_sql	kehujingyingbudb.ml_feature_add_yw_user_m								
user_status	用户状态	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m						是		
last_stop_date	最后停机时间	text	日期型	原始	kehujingyingbudb.ml_feature_info_yw_user_m							删除	否
monthsaready_last_stop_date	最后停机时间：已发生时长	numeric	数值型	手动衍生_py	python				current_date - last_stop_date				
monthsremain_last_stop_date	最后停机时间：剩余时长	numeric	数值型	手动衍生_py	python				last_stop_date - current_date				
dinner	主套餐	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
dinner_fee	套餐月租费	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_last_defer	流量滚存资源	bigint	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m					不参与当月自动衍生			
gprs_resource	GPRS国内资源总量	bigint	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_now_rest	当月GPRS资源余量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
call_use	语音资源使用量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
if_nolimit	是否不限量套餐到达用户	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
if_5g_dinner	是否5G套餐	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
if_5g_term	是否5G终端客户	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
if_jt	是否集团成员	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
start_level	客户星级标识	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
cred_type	机主证件类型	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m		不可用						
if_cred_multi	一证N号标识	bigint	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
arpu	当月ARPU	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
pay_cnt	自充值次数	bigint	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
pay_fee	自充值额度	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
acct_balance	账户余额	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
term_model	终端品牌类型	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
term_brand	终端品牌	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
term_type	终端类型	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
sk_type	双卡类型	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m					不参与自动衍生			
if_new_term	是否新机用户	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
calling_cnt	主叫通话次数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
calling_dura	当月主叫通话时长	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
called_cnt	被叫通话次数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
called_dura	当月被叫通话时长	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m		不可用						
calling_diff_cnt	主叫异网通话次数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
calling_diff_dura	主叫异网通话分钟数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_flow	GPRS总流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_flow_tnet	GPRS-T网-流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m	不可用							
gprs_flow_4g	GPRS4G流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_flow_5g	GPRS 5G流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_flow_busy	GPRS-忙时-流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_flow_idle	GPRS-闲时-流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_now_defer	可延递流量资源总量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_flow_roam	GPRS-国内漫游-流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
call_dura_roam	省际漫游-时长	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
days_roam	国内漫出天数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m		不可用			不参与自动衍生		删除	
days_roam_p	国内漫出天数/当月天数	numeric	数值型	手动衍生_py	python				days_roam / days_month				
gprs_flow_gat	GPRS-港澳台漫游-流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
days_gat	港澳台漫游-时长	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
days_outside	当月境外漫游天数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m	不可用				不参与自动衍生		删除	
days_outside_p	当月境外漫游天数占比	numeric	数值型	手动衍生_py	python				days_outside / days_month 				
days_gprs	上网天数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m					不参与自动衍生		删除	
days_gprs_p	上网天数占比	numeric	数值型	手动衍生_py	python				days_gprs / days_month				
days_call	通话天数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m					不参与自动衍生		删除	
days_call_p	通话天数占比	numeric	数值型	手动衍生_py	python				days_call / days_month				
days_call_p_1	通话天数占比_1	numeric	数值型	手动衍生_py	python				days_call_p + days_gprs				
days_call_p_2	通话天数占比_2	numeric	数值型	手动衍生_py	python				days_call_p_1 + days_gprs				
days_call_p_3	通话天数占比_3	numeric	数值型	手动衍生_py	python				days_call_p_2 + days_gprs				
days_call_p_4	通话天数占比_4	numeric	数值型	手动衍生_py	python				days_call_p_3 + days_gprs		是		
nos_call	用户通信对端手机号码数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
nos_calling	主叫通话对端号码个数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
nos_calling_diff	主叫异网对端号码个数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
sms_cnt	短信-点对点短信-条数	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
call_fee_local	本地通话费	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
call_fee_roam	漫游通话费	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_income	流量出账收入	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_fee	流量出账-流量费	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
call_fee	套餐外语音费	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_flow_video	视频类应用流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m								
gprs_flow_short	小视频系APP流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m						是		
gprs_flow_music	音乐类应用流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m						是		
gprs_flow_commu	通信类应用流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m						是		
gprs_flow_game	游戏类应用流量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_m						是		
greatest_gprs_app	app偏好	text	类别型	手动衍生_py	python				 {'gprs_flow_video': 'video', 'gprs_flow_short': 'short', 'gprs_flow_music': 'music', 'gprs_flow_commu': 'commu', 'gprs_flow_game': 'game'}		是		
paste_dinner_innet_months	主套餐、入网时长：交叉	numeric	类别型	手动衍生_py	python				(dinner, innet_months)				
dayvalue_calling_dura	次月dd日前主叫时长	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_day		不可用	不可用					
dayvalue_gprs_flow	次月dd日前流量使用量	numeric	数值型	原始	kehujingyingbudb.ml_feature_info_yw_user_day		不可用	不可用					
dayvalue_user_status	次月dd日用户状态	text	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_day		不可用	不可用					
dayvalue_phone_no_null	次月dd日	numeric	类别型	原始	kehujingyingbudb.ml_feature_info_yw_user_day		不可用	不可用					
valuecnt__days_outside_p~1	近n月出账月份数	integer	数值型	自动衍生_py	python								
flag_eg	是否eg,模型示例的目标字段	integer	数值型	原始	kehujingyingbudb.ml_target_info_yw_user_m							删除	
flag_eg2	是否eg2,模型示例2的目标字段	text	类别型	原始	kehujingyingbudb.ml_target_info_yw_user_m							删除	
score_flag_eg	模型示例的分数	numeric	数值型	原始	kehujingyingbudb.table_score_month							删除	
ago_score_flag_eg	模型示例的分数(历史)	numeric	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
score_flag_eg2	模型示例2的分数	numeric	数值型	原始	kehujingyingbudb.table_score_month							删除	
ago_score_flag_eg2	模型示例2的分数(历史)	numeric	数值型	手动衍生_py	python				{'notago_tovalue': 1}				
rn	随机排序字段	bigint	数值型	手动衍生_sql	sql					不参与自动衍生		删除	
acct_day	日表账期字段	text	类别型	其他	kehujingyingbudb.ml_feature_info_yw_user_day								
"""
# </editor-fold> -------------------------------------------------------------------------------------------------------

