

def get_inclination(file_name):
    """
    傾き
    """
    intensity = spec[spec['spectrum_filename']==file_name]['light_intensity'].values[0]
    wave = spec[spec['spectrum_filename']==file_name]['wave_length'].values[0]

    return (max(intensity) - min(intensity)) / (max(wave) - min(wave))


def generate_stats_featuers(df, col):
    df[f'{col}_max'] = df[col].apply(lambda x: max(x))
    df[f'{col}_min'] = df[col].apply(lambda x: min(x))
    df[f'{col}_std'] = df[col].apply(lambda x: np.std(x))
    df[f'{col}_mean'] = df[col].apply(lambda x: np.mean(x))
    df[f'{col}_skew'] = df[col].apply(lambda x: pd.Series(x).skew())
    df[f'{col}_kurt'] = df[col].apply(lambda x: pd.Series(x).kurt())
    new_cols = [f'{col}_max', f'{col}_min', f'{col}_std', f'{col}_mean', f'{col}_skew', f'{col}_kurt']
    return df, new_cols


# 光強度の一階微分、二階微分の差
def generate_diff(x, n):
    diff = np.diff(spec_meta_df[spec_meta_df['spectrum_filename']==x]['light_intensity'].values[0], n=n, axis=-1)

    padding_len = 512 - len(diff) 
    if padding_len < 512:
        diff = np.concatenate([diff, [0] * padding_len], 0)
    return diff


def numerical_feature_generater(x : pd.Series, mode=None):
    if mode == 'cut':
        labels, _ = pd.factorize(pd.cut(x, 4, labels=['Q1', 'Q2', 'Q3', 'Q4']))
    elif mode == 'qcut':
        labels, _ = pd.factorize(pd.qcut(x, 4, labels=['Q1', 'Q2', 'Q3', 'Q4']))
    elif mode == 'log1p':
        labels = np.log1p(x)
    return labels


if __name__ == '__main__':

    num_cols = ['col_A', 'col_B', 'col_C']
    new_cols = []
    for n_col in tqdm(num_cols):
        df_train[f'{n_col}_cut'] = numerical_feature_generater(df_train[f'{n_col}'], mode='cut')
        df_train[f'{n_col}_qcut'] = numerical_feature_generater(df_train[f'{n_col}'], mode='qcut')
        df_train[f'{n_col}_log1p'] = numerical_feature_generater(df_train[f'{n_col}'], mode='log1p')
        new_cols.append(f'{n_col}_cut')
        new_cols.append(f'{n_col}_qcut')
        new_cols.append(f'{n_col}_log1p')

    print(new_cols, len(new_cols))
    df_train.head(3)



