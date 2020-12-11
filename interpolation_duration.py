import pandas as pd
import numpy as np
from datetime import datetime, timedelta

list_symptoms = ['fatigue','abdominal_pain','chest_pain','sore_throat','shortness_of_breath',
                 'skipped_meals','loss_of_smell','unusual_muscle_pains','headache','hoarse_voice','delirium','diarrhoea',
                 'fever','persistent_cough','dizzy_light_headed','eye_soreness','red_welts_on_face_or_lips','blisters_on_feet']

list_symptoms_bin  = list_symptoms + ['sob2', 'fatigue2']


def creating_nanvalues(df_train):
    for f in list_symptoms:
        df_train[f] -= 1
        df_train[f] = np.where(df_train[f] == -1, 0,df_train[f])
    return df_train

def interpolate_date(df_train, list_symptoms=list_symptoms, col_day='interval_days'):
    df_train['sob2'] = np.round(df_train['shortness_of_breath'] * 1.0 / 3, 0)
    df_train['fatigue2'] = np.round(df_train['fatigue'] * 1.0 / 3, 0)
    df_test_comb = df_train.sort_values(col_day, ascending=False).drop_duplicates(
        [col_day, "patient_id"])
    df_test_comb = df_test_comb.sort_values(['patient_id', 'interval_days'])
    full_idx = np.arange(df_test_comb['interval_days'].min(), df_test_comb['interval_days'].max())

    for f in list_symptoms + ['fatigue', 'shortness_of_breath']:
        df_test_comb[f] = df_test_comb[f].fillna(0)
    df_test_comb_ind = df_test_comb.set_index('interval_days')

    # df_test2 = df_test_comb_ind.groupby('patient_id', as_index=False).apply(lambda group: group.reindex(full_idx)).reset_index(level=0, drop=True).sort_index()
    def f_inter(x):
        #         full_idx = np.arange(x['interval_days'].min(), x['interval_days'].max())
        x = x.reindex(full_idx)
        for f in list_symptoms:
            x[f] = x[f].replace('False', 0)
            x[f] = x[f].replace('True', 1)
            x[f] = x[f].astype(float)
            x[f] = x[f].interpolate(method='linear', limit_area='inside')
        x = x.dropna(subset=list_symptoms + ['date_update'], how='all')
        return (x)

    df_interp = df_test_comb_ind.groupby('patient_id').apply(f_inter).rename_axis(('patient_id', 'interval_days')).drop(
        'patient_id', 1).reset_index()
    return df_interp

def interpolate_healthy(dfg):
    dfg['health'] = np.nan
    dfg['health'] = np.where(dfg['sum_symp']==0,1,dfg['health'])
    dfg['health'] = np.where(dfg['sum_symp']>0,0,dfg['health'])
    dfg['health_interp'] = dfg['health'].ffill()
    dfg['health_back'] = dfg['health'].bfill()
    dfg['nan_healthy'] = np.where(dfg['health_interp']==1,np.nan,dfg['health_interp'])
    dfg['nan_uh'] = np.where(dfg['health_interp']==0,np.nan,1-dfg['health_interp'])
    dfg['count_healthy'] = dfg.nan_healthy.isnull().astype(int).groupby(dfg.nan_healthy.notnull().astype(int).cumsum()).cumsum()
    dfg['count_nans'] = dfg.health_status.isnull().astype(int).groupby(dfg.health_status.notnull().astype(int).cumsum()).cumsum()

    dfg['count_nothealthy'] = dfg.nan_uh.isnull().astype(int).groupby(dfg.nan_uh.notnull().astype(int).cumsum()).cumsum()

    dfg['nan_or_health'] = np.where(dfg['health']==0,1,np.nan)
    dfg['count_nans_or_health'] = dfg.nan_or_health.isnull().astype(int).groupby(dfg.nan_or_health.notnull().astype(int).cumsum()).cumsum()

    max_cnans = dfg['count_healthy'].max()
    idx_cnans = dfg['count_healthy'].idxmax()
    return dfg

def check_succession_healthy(dfg, limit):
    dfg['sum_nans'] = np.where(dfg['sum_symp']==0,np.nan, dfg['sum_symp'])
    dfg['count_healthy'] = dfg.sum_nans.isnull().astype(int).groupby(dfg.sum_nans.notnull().astype(int).cumsum()).cumsum()
    dfg['not_healthy'] = dfg.sum_nans.notnull().astype(int).groupby(dfg.sum_nans.isnull().astype(int).cumsum()).cumsum()
    dfg['interval_interp'] = np.arange(0,dfg.shape[0])
    dfg = dfg.reset_index(drop=True)
    dfg['interp_sum'] = dfg['sum_symp'].interpolate('linear')
    max_cnans = dfg['count_healthy'].max()
    idx_cnans = dfg['count_healthy'].idxmax()
#     dfg['interval_interp'] = dfg['interval_days'].interpolate(method='linear',limit_area='inside')
    dfg['ln'] = dfg['count_healthy'].ge(limit)
    dfg['days_max'] = 0
    value_max_e = dfg.sort_values(['sum_symp','interval_interp'], ascending=False).head(1)['interval_interp']
    value_max_b = dfg.sort_values(['sum_symp'], ascending=False).head(1)['interval_interp']

    dfg['days_max_b'] = np.asarray(value_max_b)[0]
    dfg['days_max_e'] = np.asarray(value_max_e)[0]
    print(value_max_e, value_max_b)
    dfg = dfg.sort_values('interval_interp').reset_index(drop=True)
    #print(dfg['days_max'].min())
    dfg['lne'] = dfg['ln'].astype(float) * dfg['interval_interp'].gt(dfg['days_max_e']).astype(float) * dfg['interp_sum'].lt(2).astype(float)
    dfg['lnb'] = dfg['ln'].astype(float) * dfg['interval_interp'].lt(dfg['days_max_b']).astype(float)
    dfg_temp = dfg
    if dfg['lnb'].max() >0:
        print("begin issue")
        dfg_lnb_last = dfg_temp[dfg_temp['lnb']>0].last_valid_index()
        print(np.asarray(dfg_temp.loc[dfg_lnb_last]['interp_sum']))
        dfg_temp = dfg_temp.loc[dfg_lnb_last:]
        dfg_temp = dfg_temp.reset_index(drop=True)
    if dfg['lne'].max() > 0:
        print("end issue")
        dfg_lne_first = dfg_temp[dfg_temp['lne']>0].first_valid_index()
#         print(dfg_temp.iloc[dfg_temp['lne'].idxmax()]['interval_interp'])
        dfg_temp = dfg_temp[dfg_temp['interval_interp']<dfg_temp.iloc[dfg_temp['lne'].idxmax()]['interval_interp']-limit+2]
#         print(np.asarray(max_cnans), np.asarray(idx_cnans))
#         dfg_temp = dfg.iloc[:np.asarray(idx_cnans)-np.asarray(max_cnans)]
    print(dfg_temp.shape,dfg.shape, dfg_temp[['interval_interp','days_max_e','days_max_b','sum_symp','interp_sum','duration','ln','lnb','lne']])
    return dfg


def define_gh_noimp(dfg, gap_healthy=7):
    id_max = dfg[dfg['sum_symp'] == dfg['max_symp']]['interval_days'].min()
    id_max2 = dfg[dfg['sum_symp'] == dfg['max_symp']]['interval_days'].max()
    id_test = dfg[dfg['created_at'] <= dfg['date_effective_test']]['interval_days'].max()
    # Check if id_test
    dfg_before_test = dfg[dfg['interval_days'] <= id_test]
    dfg_after_test = dfg[dfg['interval_days'] >= id_test]
    dfg_before_max = dfg[dfg['interval_days'] <= id_max]
    dfg_after_max = dfg[dfg['interval_days'] >= id_max2]
    max_count_test_nh = dfg_before_test['count_nothealthy'].max()
    max_count_bh = dfg_before_max['count_healthy'].max()
    max_count_ah = dfg_after_max['count_healthy'].max()

    if dfg_before_test.shape[0] > 0 and dfg['pcr_standard'].max() == 1:
        if max_count_test_nh > 0:
            print('treating test', dfg_before_test.shape[0], max_count_test_nh)
            last_count_ah = dfg_after_test['count_healthy'].tail(1).to_numpy()[0]
            min_count_bh = dfg_before_test['count_healthy'].min()
            dfg['first_stuh_hg%d' % gap_healthy] = dfg_before_test[dfg_before_test['count_nothealthy'] == 1][
                'interval_days'].min()
            dfg['last_stuh_hg%d' % gap_healthy] = dfg_after_test[dfg_after_test['count_healthy'] == last_count_ah][
                                                      'interval_days'].max() - last_count_ah

            dfg['first_stuh_nhg%d' % gap_healthy] = dfg_before_test[dfg_before_test['count_nothealthy'] == 1][
                'interval_days'].min()
            dfg['last_stuh_nhg%d' % gap_healthy] = dfg_after_test[dfg_after_test['count_healthy'] == last_count_ah][
                                                       'interval_days'].max() - last_count_ah

            dbm_hg = dfg_before_test[dfg_before_test['count_healthy'] > gap_healthy]
            dam_hg = dfg_after_test[dfg_after_test['count_healthy'] > gap_healthy]

            last_healthy_gap = dbm_hg['interval_days'].max()
            first_healthy_gap = dam_hg['interval_days'].min() - gap_healthy
            #     print(last_healthy_gap)
            if dbm_hg.shape[0] > 0:
                dfg['first_stuh_hg%d' % gap_healthy] = last_healthy_gap + 1
            if dam_hg.shape[0] > 0:
                dfg['last_stuh_hg%d' % gap_healthy] = first_healthy_gap

            dbm_hg = dfg_before_test[dfg_before_test['count_nans_or_health'] > gap_healthy]
            dam_hg = dfg_after_test[dfg_after_test['count_nans_or_health'] > gap_healthy]
            last_healthy_gap = dbm_hg['interval_days'].max()
            first_healthy_gap = dam_hg['interval_days'].min() - gap_healthy
            #     print(last_healthy_gap)
            if dbm_hg.shape[0] > 0:
                dfg['first_stuh_nhg%d' % gap_healthy] = last_healthy_gap + 1
            if dam_hg.shape[0] > 0:
                dfg['last_stuh_nhg%d' % gap_healthy] = first_healthy_gap
    else:
        last_count_ah = dfg_after_max['count_healthy'].tail(1).to_numpy()[0]
        min_count_bh = dfg_before_max['count_healthy'].min()
        dfg['first_stuh_hg%d' % gap_healthy] = dfg_before_max[dfg_before_max['count_healthy'] == min_count_bh][
            'interval_days'].max()
        dfg['last_stuh_hg%d' % gap_healthy] = dfg_after_max[dfg_after_max['count_healthy'] == last_count_ah][
                                                  'interval_days'].max() - last_count_ah

        dfg['first_stuh_nhg%d' % gap_healthy] = dfg_before_test[dfg_before_test['count_nothealthy'] == 1][
            'interval_days'].min()
        dfg['last_stuh_nhg%d' % gap_healthy] = dfg_after_test[dfg_after_test['count_healthy'] == last_count_ah][
                                                   'interval_days'].max() - last_count_ah

        dbm_hg = dfg_before_max[dfg_before_max['count_healthy'] > gap_healthy]
        dam_hg = dfg_after_max[dfg_after_max['count_healthy'] > gap_healthy]
        last_healthy_gap = dbm_hg['interval_days'].max()
        first_healthy_gap = dam_hg['interval_days'].min() - gap_healthy
        #     print(last_healthy_gap)
        if dbm_hg.shape[0] > 0:
            dfg['first_stuh_hg%d' % gap_healthy] = last_healthy_gap + 1
        if dam_hg.shape[0] > 0:
            dfg['last_stuh_hg%d' % gap_healthy] = first_healthy_gap

        dbm_hg = dfg_before_max[dfg_before_max['count_nans_or_health'] > gap_healthy]
        dam_hg = dfg_after_max[dfg_after_max['count_nans_or_health'] > gap_healthy]
        last_healthy_gap = dbm_hg['interval_days'].max()
        first_healthy_gap = dam_hg['interval_days'].min() - gap_healthy
        #     print(last_healthy_gap)
        if dbm_hg.shape[0] > 0:
            dfg['first_stuh_nhg%d' % gap_healthy] = last_healthy_gap + 1
        if dam_hg.shape[0] > 0:
            dfg['last_stuh_nhg%d' % gap_healthy] = first_healthy_gap
    return dfg

def date_range(dfg, c='date_update'):
    dfg['interval'] = dfg[c] - dfg[c].min()
    return dfg

def creating_interpolation(df_init):
    df_init['first_entry_sum'] = df_init.groupby('patient_id')['sum_symp'].transform('first')
    df_init['last_entry_sum'] = df_init.groupby('patient_id')['sum_symp'].transform('last')
    df_init['date_update'] = pd.to_datetime(df_init['created_at_day'].str.strip('b'),infer_datetime_format=True)
    print('Applying date range')
    df_init = df_init.groupby('patient_id').apply(date_range)
    df_init['interval_days'] = df_init['interval'].apply(lambda x: x.days)
    print('Performing interpolation')
    df_interp = interpolate_date(df_init)
    return df_interp


def creating_duration_healthybased(df_interp,days=[7],hi=False):
    if hi==True:
        df_interp['health_status'] = np.nan
        df_interp['health_status'] = np.where(df_interp['sum_symp'] > 0, 0, df_interp['health_status'])
        df_interp['health_status'] = np.where(df_interp['sum_symp'] == 0, 1, df_interp['health_status'])
        df_interp['max_symp'] = df_interp.groupby('patient_id')['sum_symp'].transform('max')
        df_interp = df_interp.groupby('patient_id').apply(lambda group: interpolate_healthy(group))

    for f in days:
        if 'first_stuh_hg%d'%f not in df_interp.columns:
            print('treating ', f)

            df_interp = df_interp.groupby('patient_id').apply(lambda group: define_gh_noimp(group, gap_healthy=f))
            df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')


    return df_interp


def determine_meeting_criteria(df_interp, days=[7]):
    from datetime import datetime, timedelta

    timemin = timedelta(days=-7)
    import time

    struct_time = datetime.strptime("20 Jul 20", "%d %b %y").date()
    df_interp['delay'] = pd.to_datetime(df_interp['date_update'], infer_datetime_format=True).dt.date - struct_time
    df_interp['dropped'] = np.where(df_interp['delay'] <= timemin, 1, 0)
    df_interp['last_entry_sum'] = df_interp.groupby('patient_id')['sum_symp'].transform('last')

    df_interp['count_uh_nans'] = df_interp['count_nans'] * (1 - df_interp['health_interp']) * (
            1 - df_interp['health_back'])
    df_interp['count_utoh_nans'] = df_interp['count_nans'] * (1 - df_interp['health_interp']) * df_interp['health_back']
    df_interp['count_htouh_nans'] = df_interp['count_nans'] * (df_interp['health_interp']) * (
            1 - df_interp['health_back'])
    df_interp['max_uh_nans'] = df_interp.groupby('patient_id')['count_uh_nans'].transform('max')

    for f in days:
        if 'postcrit_aok%d'%f not in df_interp.columns:
            print('Need to treat ', f)
            timemin = timedelta(days=-f)
            df_interp['meeting_post_criteria%d' % f] = np.where(
                np.logical_and(df_interp['interval_days'] > df_interp['last_stuh_hg%d' % f],
                               df_interp['count_healthy'] == f), 1, 0)
            df_interp['postcrit_ok%d' % f] = df_interp.groupby('patient_id')['meeting_post_criteria%d' % f].transform('max')
            df_interp['postcrit_aok%d' % f] = df_interp['postcrit_ok%d' % f]
            df_interp['postcrit_aok%d' % f] = np.where(
                np.logical_and(df_interp['delay'] < timemin, df_interp['last_entry_sum'] == 0), 1,
                df_interp['postcrit_aok%d' % f])
            df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')

    for f in list_symptoms:
        df_interp['check_' + f] = np.where(df_interp['health_interp'] == 0, df_interp[f], 0)
        df_interp['sumcheck_' + f] = df_interp.groupby('patient_id')['check_' + f].transform('sum')

    for d in days:
        df_interp['sick%d' % d] = np.where(
            np.logical_and(df_interp['interval_days'] >= df_interp['first_stuh_hg%d' % d],
                           df_interp['interval_days'] <= df_interp['last_stuh_hg%d' % d]), 1, 0)
        for f in list_symptoms:
            df_interp['sick%d_' % d + f] = np.where(df_interp['sick%d' % d] == 1, df_interp['check_' + f], 0)
            df_interp['day%d_' % d + f] = np.where(
                np.logical_and(df_interp['sick%d' % d] == 1, df_interp['check_' + f] > 0.5), df_interp['interval_days'],
                np.nan)
            df_interp['start%d_' % d + f] = df_interp.groupby('patient_id')['day%d_' % d + f].transform('min')
            df_interp['sumsick%d_' % d + f] = df_interp.groupby('patient_id')['sick%d_' % d + f].transform('sum')
    return df_interp

# df_init = pd.read_csv('/home/csudre/MountedSpace/Covid/PositiveSympStartHealthyAllSymptoms.csv')
# for c in df_init.columns:
#     print(c)
# df_init = creating_nanvalues(df_init)
# df_interp = creating_interpolation(df_init)
# print('Saving interpolated ')
# df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')

print('loading file')
df_test = pd.read_csv('/home/csudre/MountedSpace/Covid/TestedPositiveTestDetails.csv')
df_interp = pd.read_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')

df_test_pos = df_test[df_test['result']==1]
df_test_min = df_test.sort_values(['date_effective_test','patient_id']).drop_duplicates('patient_id')
df_interp = pd.merge(df_interp,df_test_min,on='patient_id',how='left')
df_interp = df_interp.drop(columns=['first_stuh_hg7'])
print('processing')
df_interp = creating_duration_healthybased(df_interp,hi=False)
df_interp = determine_meeting_criteria(df_interp)
print('sving meeing crite')
df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp2.csv')
# Checking hosp
print('Checking hosp')
# df_interp = pd.read_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp.csv')
# print(df_interp.shape)
df_interp['hosp_valid'] = (df_interp['interval_days']>df_interp['first_stuh_hg7']) * (df_interp['location']>1)
df_interp['hosp_check'] = df_interp.groupby('patient_id')['hosp_valid'].transform('max')
#
df_interp.to_csv('/home/csudre/MountedSpace/Covid/InterpolatedPosSHSymp2.csv')
df_unique = df_interp.sort_values(['patient_id','sum_symp'],ascending=False).drop_duplicates('patient_id')
df_pat = pd.read_csv('/home/csudre/MountedSpace/Covid/PositiveSympStartHealthy_PatDetails.csv')

df_merge = pd.merge(df_unique,df_pat, left_on='patient_id',right_on='id')
df_merge['age'] = 2020 - df_merge['year_of_birth']
df_merge.to_csv('/home/csudre/MountedSpace/Covid/UniqueForDuration.csv')
for f in list_symptoms:
    df_interp[f] = np.where(df_interp['health_interp']==1,0,df_interp[f])

df_interp_firstweek = df_interp[(df_interp['interval_days']>=df_interp['first_stuh_hg7']) & (df_interp['interval_days']<df_interp['first_stuh_hg7']+8) & (df_interp['last_stuh_hg7']-df_interp['first_stuh_hg7']+1>=7)]
df_interp_firstweek_uk = df_interp_firstweek[df_interp_firstweek['country_code']=="b'GB'"]
df_interp_firstweek_uk['rednans']  = np.where(df_interp_firstweek_uk['health'].isna(),1,0)
df_interp_firstweek_uk['count_rednans'] = df_interp_firstweek_uk.groupby('patient_id')['rednans'].transform('sum')
df_interp_firstweek_ukfin = df_interp_firstweek_uk[df_interp_firstweek_uk['count_rednans']<4]
df_interp_firstweek_ukfin.to_csv('/home/csudre/MountedSpace/Covid/FirstWeekData.csv')

print(len(list(set(df_interp_firstweek_ukfin['patient_id']))))