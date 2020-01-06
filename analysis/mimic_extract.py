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
                           .fillna(0))
    df_out.loc[:,idx[:,'mean']] = imputed_means

    mask = (df.loc[:, idx[:,'count']] > 0).astype(float)
    df_out.loc[:,idx[:, 'count']] = mask
    df_out.rename(columns={'count': 'mask'}, 
                  level='Aggregation Function', 
                  inplace=True)
    
    is_absent = (1 - df_out.loc[:, idx[:,'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = (hours_of_absence 
                           - (hours_of_absence[is_absent==0]
                                .fillna(method='ffill')))
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, 
                               level='Aggregation Function', 
                               inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    time_since_measured = (df_out.loc[:, idx[:, 'time_since_measured']]
                                .fillna(WINDOW_SIZE+1))
    df_out.loc[:, idx[:, 'time_since_measured']] = time_since_measured
    df_out.sort_index(axis=1, inplace=True)
    return df_out

def extract(target, random_seed):

    statics = pd.read_hdf("data/all_hourly_data.h5", 'patients')
    data_full_lvl2 = pd.read_hdf("data/all_hourly_data.h5", 'vitals_labs')

    statics = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME]
    Ys = statics[['mort_hosp', 'mort_icu', 'los_icu']]
    Ys['los_3'] = Ys['los_icu'] > 3
    Ys['los_7'] = Ys['los_icu'] > 7
    Ys.drop(columns=['los_icu'], inplace=True)

    lvl2 = data_full_lvl2[(data_full_lvl2
                    .index.get_level_values('icustay_id')
                    .isin(set(Ys.index.get_level_values('icustay_id')))) &
                (data_full_lvl2
                    .index.get_level_values('hours_in') < WINDOW_SIZE)] 

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
        [df[df.index.get_level_values('subject_id').isin(s)] 
            for s in (train_subj, test_subj)] for df in (lvl2, Ys)]

    idx = pd.IndexSlice
    lvl2_means = lvl2_train.loc[:,idx[:,'mean']].mean(axis=0)
    lvl2_stds = lvl2_train.loc[:,idx[:,'mean']].std(axis=0)

    vals_norm = (lvl2_train.loc[:,idx[:,'mean']] - lvl2_means)/lvl2_stds
    lvl2_train.loc[:,idx[:,'mean']] = vals_norm
    vals_norm = (lvl2_test.loc[:,idx[:,'mean']] - lvl2_means)/lvl2_stds
    lvl2_test.loc[:,idx[:,'mean']] = vals_norm

    lvl2_train, lvl2_test = [simple_imputer(df) 
                                for df in (lvl2_train, lvl2_test)]
    lvl2_flat_train, lvl2_flat_test = [(df
                .pivot_table(index=['subject_id', 'hadm_id', 'icustay_id'], 
                                               columns=['hours_in'])) 
                                   for df in (lvl2_train, lvl2_test)]


    return (lvl2_flat_train.values, 
            lvl2_flat_test.values,
            Ys_train[target].values,
            Ys_test[target].values)





