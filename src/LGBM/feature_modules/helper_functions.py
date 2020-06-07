from numba import jit
from numpy.fft import *
from scipy import fftpack


def padding(x):
    """
    可変系列長の波形データを padding する.
    """
    len_padding = 512 - len(x)
    if len_padding > 0:
        x = np.concatenate([x, [0] * len_padding], 0)
    return x


def get_peak_around(x, half_num_around):
    """
    波形データの peak 周辺のデータを切り出す.
    """
    i = np.argmax(x)
    if i < half_num_around:
        return x[:half_num_around*2]
    elif i+half_num_around > 512:
        return x[-(half_num_around*2):]

    return x[i - half_num_around:i + half_num_around]


@jit('float32(float32[:,:], int32)')
def feature_extractor(x, n_part=1000):
    """
    電線コンペで見つけたやつ.
    波形データの特徴点を抽出する.
    """
    lenght = len(x)
    pool = np.int32(np.ceil(lenght/n_part))
    output = np.zeros((n_part,))
    for j, i in enumerate(range(0,lenght, pool)):
        if i+pool < lenght:
            k = x[i:i+pool]
        else:
            k = x[i:]
        output[j] = np.max(k, axis=0) - np.min(k, axis=0)
    return output


#FFT to filter out HF components and get main signal profile
def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)


###Filter out low frequencies from the signal to get HF characteristics
def high_pass(s, threshold=1e7):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies < threshold] = 0
    return irfft(fourier)


# ピーク付近の和
def peak_near_sum(x):
    i = np.argmax(x)
    z = x[i - 10:i + 10]
    return np.sum(z) / x[i]


def merge_extracted_listed_feature(df, listed_col_name):
    tmp = pd.DataFrame(np.stack(df[listed_col_name]), 
                       columns=[f'{listed_col_name}_{i}' for i in range(len(np.stack(df[listed_col_name])[0]))])
    new_cols = list(tmp.columns)
    df = pd.concat([df, tmp], 1)
    return df, new_cols


def compress_feature_generator(df, col_name):
    new_columns = []
    n_components = 5
    pca = PCA(n_components=n_components)
    X_PCA = pca.fit_transform(np.stack(df[col_name]))
    X_PCA = pd.DataFrame(X_PCA, columns=[f'PCA_5d_{i}_{col_name}' for i in range(n_components)])
    new_columns.extend(list(X_PCA.columns))

    n_clusters = 4
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=10).fit(np.stack(df[col_name]))
    k_labels = kmeans_model.labels_
    new_columns.extend([f'k{n_clusters}_means_cluster_{col_name}'])

    nmf = NMF(n_components=n_components)
    W_NMF = nmf.fit_transform(abs(np.stack(df[col_name])))
    W_NMF = pd.DataFrame(W_NMF, columns=[f'NMF_5d_{i}_{col_name}' for i in range(n_components)])
    new_columns.extend(list(W_NMF.columns))

    df[f'k{n_clusters}_means_cluster_{col_name}'] = k_labels
    df = pd.concat([df, X_PCA], 1)
    df = pd.concat([df, W_NMF], 1)
    return df, new_columns
