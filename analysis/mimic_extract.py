import pandas as pd
import numpy as np
import pickle

GAP_TIME          = 6  # In hours
WINDOW_SIZE       = 24 # In hours
ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']
TRAIN_FRAC, TEST_FRAC = 0.7, 0.3

def simple_imputer(df):
    idx = pd.IndexSlice
    df = df.copy()
    if len(df.columns.names) > 2: 
        df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))
    
    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    imputed_means = (df_out.loc[:,idx[:,'mean']]
                           .groupby(ID_COLS).fillna(method='ffill')
                           .groupby(ID_COLS).fillna(icustay_means)
                           .fillna(0)).copy()
    df_out.loc[:,idx[:,'mean']] = imputed_means

    mask = (df.loc[:, idx[:,'count']] > 0).astype(float).copy()
    df_out.loc[:,idx[:,'count']] = mask
    df_out = df_out.rename(columns={'count': 'mask'}, 
                            level='Aggregation Function')
   
    is_absent = (1 - df_out.loc[:, idx[:,'mask']].copy())
    hours_of_absence = is_absent.cumsum()
    time_since_measured = (hours_of_absence 
                           - (hours_of_absence[is_absent==0]
                                .fillna(method='ffill')))
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, 
                               level='Aggregation Function', 
                               inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    time_since_measured = (df_out.loc[:, idx[:, 'time_since_measured']]
                                .fillna(WINDOW_SIZE+1)).copy()
    df_out.loc[:, idx[:, 'time_since_measured']] = time_since_measured
    df_out.sort_index(axis=1, inplace=True)
    return df_out

def extract(random_seed, target):

    statics = pd.read_hdf("data/all_hourly_data.h5", 'patients')
    data_full_lvl2 = pd.read_hdf("data/all_hourly_data.h5", 'vitals_labs')

    statics = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME]
    Ys = statics.loc[:,['mort_hosp', 'mort_icu', 'los_icu']]
    Ys.loc[:,"mort_hosp"] = (Ys.loc[:,"mort_hosp"]).astype(int)
    Ys.loc[:,"mort_icu"] = (Ys.loc[:,"mort_icu"]).astype(int)
    Ys.loc[:,'los_3'] = (Ys.loc[:,'los_icu'] > 3).astype(int)
    Ys.loc[:,'los_7'] = (Ys.loc[:,'los_icu'] > 7).astype(int)
    Ys.drop(columns=['los_icu'], inplace=True)

    lvl2 = data_full_lvl2.loc[(data_full_lvl2
                    .index.get_level_values('icustay_id')
                    .isin(set(Ys.index.get_level_values('icustay_id')))) &
                (data_full_lvl2
                    .index.get_level_values('hours_in') < WINDOW_SIZE),:] 

    lvl2_subj_idx, Ys_subj_idx = [df.index.get_level_values('subject_id') 
                                    for df in (lvl2, Ys)]
    lvl2_subjects = set(lvl2_subj_idx)
    assert lvl2_subjects == set(Ys_subj_idx), "Subject ID pools differ!"

    np.random.seed(random_seed)
    subjects = np.random.permutation(list(lvl2_subjects))
    N = len(lvl2_subjects)
    N_train, N_test = int(TRAIN_FRAC * N), int(TEST_FRAC * N)
    train_subj = subjects[:N_train]
    test_subj  = subjects[N_train:]

    [(lvl2_train, lvl2_test), (Ys_train, Ys_test)] = [
        [df.loc[df.index.get_level_values('subject_id').isin(s),:] 
            for s in (train_subj, test_subj)] for df in (lvl2, Ys)]

    idx = pd.IndexSlice
    lvl2_means = lvl2_train.loc[:,idx[:,'mean']].mean(axis=0)

    # here, we do not use the standard deviation. 
    # Tree models do not get affected by the scale of values.
    lvl2_stds = lvl2_train.loc[:,idx[:,'mean']].std(axis=0)

    vals_centered = (lvl2_train.loc[:,idx[:,'mean']] - lvl2_means)
    lvl2_train.loc[:,idx[:,'mean']] = vals_centered
    vals_centered = (lvl2_test.loc[:,idx[:,'mean']] - lvl2_means)
    lvl2_test.loc[:,idx[:,'mean']] = vals_centered

    lvl2_train, lvl2_test = [simple_imputer(df) 
                                for df in (lvl2_train, lvl2_test)]

    lvl2_flat_train, lvl2_flat_test = [(df
                .pivot_table(index=['subject_id', 'hadm_id', 'icustay_id'], 
                                               columns=['hours_in'])) 
                                   for df in (lvl2_train, lvl2_test)]
    

    return (lvl2_flat_train.values, 
            lvl2_flat_test.values,
            Ys_train.loc[:,target].values,
            Ys_test.loc[:,target].values)





