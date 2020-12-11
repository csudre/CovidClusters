import pandas as pd
import numpy as np
import glob
import argparse
import sys

list_symptoms = ['abdominal_pain','chest_pain','sore_throat','sob2','fatigue2','headache','hoarse_voice','loss_of_smell',
          'delirium','diarrhoea','fever','persistent_cough','unusual_muscle_pains','skipped_meals']


def individual_projection(df, proj, list_patient):
    list_pat = np.unique(list_patient)
    print(proj.shape, len(list_patient), len(list_pat))
    sst = np.matmul(np.asarray(proj), proj.T)
    list_projection = []
    for patient in list_pat:
        # print(patient)
        # print(df.loc[patient])
        projected = np.matmul(np.asarray(df.loc[patient]), proj)
        list_projection.append(projected[-1, :])
    return np.vstack(list_projection)


def error_calculation(sst, array, drop='all'):
    # sst = np.matmul(np.asarray(proj), proj.T)
    # array_new = array.interpolate(method='linear', limit_area='inside')
    array_new = array
    if drop == 'any':
        array_new = array_new.dropna(how='any')
    else:
        array_new = array_new.dropna(how='all')
        array_new = array_new.fillna(0)
    # print(array_new.shape,array.shape)
    projected = np.matmul(np.asarray(array_new), sst)
    # print(array, projected)
    error = np.sum(np.square(np.asarray(array_new - projected)))
    # print("error is ", error)
    return error


def create_dict_cov(df):
    dict_cov = {}
    list_patients = df.index.levels[0]
    for p in list_patients:
        dict_cov[p] = np.nan_to_num(cov_patient(df, p, norm=True, drop='all'))
    return dict_cov

def projection_cluster_cov(dict_cov, patient_list, numb_dim=2):
    if len(patient_list) == 0:
        print("no patient in the list")
        return None
    cov_fin = np.zeros(dict_cov[patient_list[0]].shape)
    # print(cov_fin.shape)

    for patient in patient_list:
        cov_temp = dict_cov[patient]
        # print(cov_temp)
        cov_fin += cov_temp
    cov_fin /= len(patient_list)
    # print(cov_fin)
    u, s, v = np.linalg.svd(cov_fin)
    #     print(u.shape,s.shape,v.shape)
    #     print(np.asarray(u)[:,:numb_dim].shape, u.shape,numb_dim)
    u_array = np.asarray(u)
    #     print(u_array.shape, numb_dim, 'u_array shape')
    return u_array[:, :numb_dim]


def projection_cluster(df, patient_list, numb_dim=2, drop='all'):
    cov_fin = np.zeros([len(df.columns), len(df.columns)])
    # print(cov_fin.shape)
    if len(patient_list) == 0:
        print("no patient in the list")
        return cov_fin
    for patient in patient_list:
        cov_temp = cov_patient(df, patient, drop=drop)
        cov_temp = np.nan_to_num(cov_temp)
        # print(cov_temp)
        cov_fin += cov_temp
    cov_fin /= len(patient_list)
    # print(cov_fin)
    u, s, v = np.linalg.svd(cov_fin)
    #     print(u.shape,s.shape,v.shape)
    #     print(np.asarray(u)[:,:numb_dim].shape, u.shape,numb_dim)
    u_array = np.asarray(u)
    #     print(u_array.shape, numb_dim, 'u_array shape')
    return u_array[:, :numb_dim]


def cov_patient(df, patient, norm=True, drop='all'):
    test_init = df.loc[patient].interpolate(method='linear',
                                            limit_area='inside')
    if drop == 'any':
        test_0 = test_init.dropna(how='any')
    else:
        test_0 = test_init.dropna(how='all')
        test_0 = test_init.fillna(0)
    # test_norm = (test_0 - test_0.mean()) / test_0.std() + (test_0.mean()) / (
    #         test_0.max() - test_0.min()) * test_0.max()
    test_norm = test_0 - test_0.mean()
    if norm:
        cov = np.cov(np.asarray(test_norm).T)
    else:
        cov = np.cov(np.asarray(test_0).T)
    return cov


def LB_Keogh(s1, s2, r):
    LB_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound) ** 2

    return np.sqrt(LB_sum)


def DTWDistance(s1, s2, w):
    DTW = {}
    s1b = np.asarray(s1[~np.isnan(s1)])
    s2b = np.asarray(s2[~np.isnan(s2)])
    # print(s1b, s2b, s1, s2)
    w = np.max([w, abs(len(s1) - len(s2))])
    # print(w)

    for i in range(-1, len(s1b)):
        for j in range(-1, len(s2b)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1b)):
        for j in range(max(0, i - w), min(len(s2b), i + w)):
            dist = (s1b[i] - s2b[j]) ** 2
            DTW[(i, j)] = dist + np.min(
                [DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]])

    return np.sqrt(DTW[len(s1b) - 1, len(s2b) - 1])

def within_cluster_error_calculation(df, classif, num_dim=5,):
    list_wss = []
    for f in classif.keys():
        print(f, 'taking care of cluster')
        projection = projection_cluster(df,classif[f],num_dim)
        error = 0
        for p in classif[f]:
            error +=  error_calculation(projection, df.loc[p])
        list_wss.append(error/len(classif[f]))
    return list_wss

def per_patient_error(df, list_classif, patient, num_dims=5):
    projection = projection_cluster(df,classif[f],num_dim)
    error = error_calculation(projection, df.loc[patient])
    return error


list_pat = ['000eacc47640d000c968fa3391c052ca',
            '00fdb85d4cc251fb50ca893d9271caa5',
            '0065114efaa430de58b05eb732dc32f3',
            '039cc32cdcf792e84f7a89df3eed7f66',
            '0d6acec7c378ffea6c222492a849a0f3']

def mc2pca_cov(dict_cov, df, num_clust=5, p=2, num_iter=20, random_seed=33,
               list_pat=None,
           drop='all'):
    np.random.seed(random_seed)
    if random_seed is None:
        num_int = np.arange(0, len(df.index.levels[0])) % num_clust
    else:
        num_int = np.random.randint(0, high=num_clust,
                                    size=len(df.index.levels[0]), dtype='l')
    # print(num_int)
    centroid_init = []

    for f in range(0, num_clust):
        centroid_init.append([])
    #     num_int = np.random.randint(0,len(df.index.levels[0]),num_clust)
    list_patients = df.index.levels[0]
    #     print('number of patients is ', len(list_patients), len(list(set(list_patients))))
    #     centroid_init = df.index.levels[0][num_int]
    #        num_int = np.random.randint(0,len(df.index.levels[0]),num_clust-len(centroid_init))
    if list_pat is not None:
        for (f, pat) in enumerate(list_pat):
            centroid_init[f].append(pat)
    else:
        for f in range(0, num_clust):
            centroid_init[f] = list_patients[np.where(num_int == f)]

    proj_centroids = {}
    for (i, f) in enumerate(centroid_init):
        proj_centroids[i] = projection_cluster_cov(dict_cov, f, p)
    #         print(proj_centroids[i])
    patient_classif = {}
    it = 0
    old_error = 1e32
    total_error = 100000000000
    while (old_error - total_error) / old_error > 0.0001 and it < num_iter + 60:
        old_error = total_error
        error_array = np.zeros([num_clust, len(list_patients)])
        for c_ind in range(num_clust):
            proj = proj_centroids[c_ind]
            sst = np.matmul(np.asarray(proj), proj.T)
            for (p_ind, patient) in enumerate(list_patients):
                error_array[c_ind, p_ind] = error_calculation(
                    sst, df.loc[patient], drop=drop)
                # print(error_array[c_ind,p_ind])
        temp_attribution = np.argmin(error_array, 0)
        total_error = np.sum(np.min(error_array, 0))
        print("Treating iteration ", it, "Error is ", total_error, " ratio is ",
              (old_error - total_error) / old_error)
        if total_error < old_error:
            for c_ind in range(num_clust):
                patient_clust = list_patients[
                    np.where(temp_attribution == c_ind)]
                proj_centroids[c_ind] = projection_cluster_cov(dict_cov,
                                                            patient_clust, p)
                patient_classif[c_ind] = patient_clust

        it += 1

    return proj_centroids, patient_classif, np.min(error_array, 0)


def mc2pca(df, num_clust=5, p=2, num_iter=20, random_seed=33, list_pat=None,
           drop='all'):
    np.random.seed(random_seed)
    if random_seed is None:
        num_int = np.arange(0, len(df.index.levels[0])) % num_clust
    else:
        num_int = np.random.randint(0, high=num_clust,
                                    size=len(df.index.levels[0]), dtype='l')
    # print(num_int)
    centroid_init = []

    for f in range(0, num_clust):
        centroid_init.append([])
    #     num_int = np.random.randint(0,len(df.index.levels[0]),num_clust)
    list_patients = df.index.levels[0]
    #     print('number of patients is ', len(list_patients), len(list(set(list_patients))))
    #     centroid_init = df.index.levels[0][num_int]
    #        num_int = np.random.randint(0,len(df.index.levels[0]),num_clust-len(centroid_init))
    if list_pat is not None:
        for (f, pat) in enumerate(list_pat):
            centroid_init[f].append(pat)
    else:
        for f in range(0, num_clust):
            centroid_init[f] = list_patients[np.where(num_int == f)]
            # print(centroid_init[f][0])
    #             print(list_patients[np.where(num_int==f)] )
    # print(centroid_init[f])
    #     print(centroid_init[0])
    proj_centroids = {}
    for (i, f) in enumerate(centroid_init):
        proj_centroids[i] = projection_cluster(df, f, p, drop=drop)
    #         print(proj_centroids[i])
    patient_classif = {}
    it = 0
    old_error = 1e32
    total_error = 100000000000
    while (old_error - total_error) / old_error > 0.0001 and it < num_iter + 10:
        old_error = total_error
        error_array = np.zeros([num_clust, len(list_patients)])
        for c_ind in range(num_clust):
            for (p_ind, patient) in enumerate(list_patients):
                error_array[c_ind, p_ind] = error_calculation(
                    proj_centroids[c_ind], df.loc[patient], drop=drop)
                # print(error_array[c_ind,p_ind])
        temp_attribution = np.argmin(error_array, 0)
        total_error = np.sum(np.min(error_array, 0))
        print("Treating iteration ", it, "Error is ", total_error, " ratio is ", (old_error-total_error)/old_error)
        if total_error < old_error:
            for c_ind in range(num_clust):
                patient_clust = list_patients[
                    np.where(temp_attribution == c_ind)]
                proj_centroids[c_ind] = projection_cluster(df, patient_clust, p)
                patient_classif[c_ind] = patient_clust

        it += 1

    return proj_centroids, patient_classif, np.min(error_array, 0)


# classif_bs = {}
# c_bs = {}
# error_bs = {}
def multiple_tries(df_interp, num_clust, num_trials, list_features, drop='all',
                   p=5,random_seed=0):
    list_c = []
    list_classif = []
    list_error = []
    for i in range(0, num_trials):
        print("Treating sampling ", i, " out of ", num_trials)
        df_select_ind = df_interp.set_index(['patient_id', 'interval_days'])
        df_select_fin = df_select_ind[list_features]
        c_, classif_, error_ = mc2pca(df_select_fin, num_clust=num_clust, p=p,
                                      num_iter=20, random_seed=i+random_seed,
                                      drop='all',
                                      list_pat=None)
        list_c.append(c_)
        list_classif.append(classif_)
        list_error.append(error_)
    return list_c, list_classif, list_error

def multiple_tries_cov(df_interp, num_clust, num_trials, list_features,
                     drop='all',
                   p=5,random_seed=0):
    list_c = []
    list_classif = []
    list_error = []
    df_select_ind = df_interp.set_index(['patient_id', 'interval_days'])
    df_select_fin = df_select_ind[list_features]
    dict_cov = create_dict_cov(df_select_fin)
    for i in range(0, num_trials):
        print("Treating sampling ", i, " out of ", num_trials)
        c_, classif_, error_ = mc2pca_cov(dict_cov,df_select_fin,
                                          num_clust=num_clust,
                                         p=p,
                                      num_iter=20, random_seed=i+random_seed,
                                          drop='all',
                                      list_pat=None)
        list_c.append(c_)
        list_classif.append(classif_)
        list_error.append(error_)
    return list_c, list_classif, list_error

def f_inter(x):
    x = x.reindex(full_idx)
#     x['pid'] = x['pid'].ffill().bfill()
#    x['hosp_bin'] = x['hosp_bin'].ffill().bfill()
#     x['cluster_knn'] = x['cluster_knn'].ffill().bfill()
    for f in list_symptoms:
        x[f] = x[f].replace('False', 0)
        x[f] = x[f].replace('True', 1)
        x[f] = x[f].astype(float)
        x[f] = x[f].interpolate(method='linear',limit_area='inside')
    return (x)
#     c_bs[n] = list_c
#     classif_bs[n] = list_classif
#     error_bs[n] = list_error

def encode_symptoms(df):
    searchfor_ump= ['achy','aching','muscle','myalgia','muscular']
    df['other_symptoms'] =df['other_symptoms'].fillna('none')
    df['other_symptoms'] = df['other_symptoms'].replace(-1,'none')
    df_ump = df[df['other_symptoms'].str.contains('|'.join(searchfor_ump))]
    df['unusual_muscle_pains'] = np.where(df['other_symptoms'].str.contains('|'.join(searchfor_ump)),1,df['unusual_muscle_pains'])
    return df

def encoding_treatment(df_final, df_unique):
    assess_hosp = df_final[df_final['hosp_bin']==1]
    df_oxy = assess_hosp[assess_hosp.treatment.isin(['oxygen'])]
    searchfor_vent = ['Venti']
    searchfor = ['iv', 'IV','Iv','Fluids','Intravenous']
    searchfor_ab= ['paracetamol','antibio','relief','Antibio','illin','pain']
    assess_hosp['treatment2'] = assess_hosp['treatment'].astype(str)
    df_oxy = assess_hosp[assess_hosp['treatment2'].str.contains('xygen')]
    df_venti = assess_hosp[assess_hosp['treatment2'].str.contains('|'.join(searchfor_vent))]
    df_iv = assess_hosp[assess_hosp['treatment2'].str.contains('|'.join(searchfor))]
    df_ab = assess_hosp[assess_hosp['treatment2'].str.contains('|'.join(searchfor_ab))]
    df_none = assess_hosp[assess_hosp['treatment2'].str.contains('none')]
    # df_iv = df_hosp[df_hosp.treatment.isin(['Iv','IV','Fluids','antibiotics'])]
    print(len(list(set(df_venti.patient_id))),len(list(set(df_iv.patient_id))),len(list(set(df_ab.patient_id))),len(list(set(df_oxy.patient_id))))
    df_unique['treatment_coded'] = -1
    df_unique['treatment_coded'] = np.where(df_unique['patient_id'].isin(df_none.patient_id),0,df_unique['treatment_coded'])

    df_unique['treatment_coded'] = np.where(df_unique['patient_id'].isin(df_ab.patient_id),1,df_unique['treatment_coded'])
    df_unique['treatment_coded'] = np.where(df_unique['patient_id'].isin(df_iv.patient_id),2,df_unique['treatment_coded'])
    df_unique['treatment_coded'] = np.where(df_unique['patient_id'].isin(df_oxy.patient_id),3,df_unique['treatment_coded'])

    df_unique['treatment_coded'] = np.where(df_unique['patient_id'].isin(df_venti.patient_id),4,df_unique['treatment_coded'])
    return df_final, df_unique


from sklearn.model_selection import KFold as KF
def cross_val(df, num_folds=3, num_secondfold=5):
    kf_ext = KF(n_splits=num_folds, shuffle=False, random_state=None)
    list_patients = df.index.levels[0]
    for train_ext, test_ext in kf_ext.split(list_patients):
        new_tosplit = list_patients[train_ext]
        list_fixed = list_patients[test_ext]
        kf_int = KF(n_splits=num_secondfold, )
        for train_int, test_int in kf_int.split(new_tosplit):

            data_totrain = df[df.index.levels[0].isin(list_ext)]



def main(argv):

    parser = argparse.ArgumentParser(description='Create evaluation file when'
                                                 ' comparing two segmentations')
    parser.add_argument('-hosp', dest='hosp', metavar='seg pattern',
                        type=str, required=False,
                        default='FinalHospData_2104.csv',
                        help='RegExp pattern for the segmentation files')
    parser.add_argument('-pos', dest='pos', action='store',
                        default='../TestedPosNonHosp_2104.csv', type=str,
                        help='RegExp pattern for the reference files')
    parser.add_argument('-full', dest='full', action='store',
                        default='TrainingLongData_3004.csv')
    parser.add_argument('-num_clust', dest='num_clust', action='store',
                        default=6,
                        type=int)
    parser.add_argument('-num_tries', dest='num_tries', action='store',
                        type=int,
                        default=15)
    parser.add_argument('-random', dest='random', action='store',
                        type=int,
                        default=15)
    parser.add_argument('-save_name', dest='save_name', action='store',
                        default='', help='name to save results')
    parser.add_argument('-no_interp', dest='no_interp', action='store_true',
                        default='', help='name to save results')
    try:
        args = parser.parse_args()
        # print(args.accumulate(args.integers))
    except argparse.ArgumentTypeError:
        print('compare_segmentation.py -s <segmentation_pattern> -r '
              '<reference_pattern> -t <threshold> -a <analysis_type> '
              '-save_name <name for saving> -save_maps  ')

        sys.exit(2)
    if args.full is None:
        df_hosp = pd.read_csv(glob.glob(args.hosp)[0])
        df_hosp['hosp_bin'] = 1
        df_nohosp = pd.read_csv(args.pos)
        df_nohosp['hosp_bin'] = 0
        df_nohosp['treatment_encoded'] = -1
        df_nohosp['treatment_coded'] = -1
        df_nohosp['days_to_hosp'] = 0
        df_final = pd.concat((df_hosp, df_nohosp))
        list_subjects_postreat = list(set(df_final[(df_final['test'] == 2) | (
                    df_final['treatment_coded'] > 0) | (df_final[
                                                             'guessed_pos'] > 0) | (
                                                                df_final[
                                                                    'classic_sum'] > 1)][
                                              'patient_id']))
        df_final = df_final[df_final['patient_id'].isin(list_subjects_postreat)]


    else:
        df_final = pd.read_csv(glob.glob(args.full)[0])
    df_unique = df_final.sort_values('interval_days',ascending=False).drop_duplicates('patient_id')
    print("Everything Merged")
    # df_comb = df_concatall_clean[list_symptoms+['patient_id','interval_days','hosp_bin','test']]
    df_final['sob2'] = np.round(df_final['shortness_of_breath']*1.0/3,0)
    df_final['fatigue2'] = np.round(df_final['fatigue']*1.0/3,0)
    df_test_comb = df_final.drop_duplicates(["interval_days", "patient_id"])

    full_idx = np.arange(df_test_comb['interval_days'].min(), df_test_comb[
        'interval_days'].max())

    def f_inter(x):
        x = x.reindex(full_idx)
        #     x['pid'] = x['pid'].ffill().bfill()
        # x['hosp_bin'] = x['hosp_bin'].ffill().bfill()
        #     x['cluster_knn'] = x['cluster_knn'].ffill().bfill()
        for f in list_symptoms:
            x[f] = x[f].replace('False', 0)
            x[f] = x[f].replace('True', 1)
            x[f] = x[f].astype(float)
            x[f] = x[f].interpolate(method='linear', limit_area='inside')
        return (x)
    #df_test_comb_ind = df_test_comb.set_index('interval_days')
    # df_test2 = df_test_comb_ind.groupby('patient_id', as_index=False).apply(lambda group: group.reindex(full_idx)).reset_index(level=0, drop=True).sort_index()
    df_interp = df_test_comb	
    #if args.no_interp:
        #df_interp = df_test_comb_ind.rename_axis(('patient_id',
        #                                          'interval_days')).drop(
        #    'patient_id',1).reset_index()
    #else:
        #df_interp = df_test_comb_ind.groupby('patient_id').apply(f_inter).rename_axis(('patient_id','interval_days')).drop('patient_id', 1).reset_index()
    print("Interpolation done")
    #df_interp = encode_symptoms(df_interp)
    df_unique = df_interp.sort_values('interval_days',
                                          ascending=False).drop_duplicates('patient_id')
    # df_interp, df_unique = encoding_treatment(df_interp,
    #                                          df_unique)
    df_error_postreat = pd.DataFrame([np.arange(0,len(list(set(df_interp[
                                                                   'patient_id'])))), list(set(df_interp['patient_id']))]).T
    df_error_postreat.columns = ['ind_pid','patient_id']
    print(df_error_postreat.shape, "Error initialised")

    c_, classif_, error_ = multiple_tries_cov(df_interp, args.num_clust,
                                          args.num_tries,
                                          list_symptoms,p=6,
                                              random_seed=args.random)
    r = args.num_clust
    t = args.num_tries
    for n in range(0, args.num_tries):
        df_error_postreat['error%d_numb%d' % (r, n)] = error_[n]
        df_error_postreat['clust%d_numb%d' % (r, n)] = 0
        for c in range(0, args.num_clust):
            df_error_postreat['clust%d_numb%d' % (r, n)] = np.where(
                df_error_postreat['patient_id'].isin(classif_[n][c]), c,
                df_error_postreat['clust%d_numb%d' % (r, n)])
    df_error_postreat.to_csv(args.save_name+'_C%dT%dR%d_'% (
        args.num_clust, args.num_tries, args.random)+'.csv' )

if __name__ == "__main__":
   main(sys.argv[1:])

