

def to_pickle(filename, obj):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)

def unpickle(filename):
    with open(filename, mode='rb') as fo:
        p = pickle.load(fo)
    return p 

def padding(x):
    len_padding = 512 - len(x)
    if len_padding > 0:
        x = np.concatenate([x, [0] * len_padding], 0)
    return x

def paddingV2(x, total_length):
    len_padding = total_length - len(x)
    if len_padding > 0:
        x = np.concatenate([x, [0] * len_padding], 0)
    return x

def get_peak_around(x, half_num_arround):
    i = np.argmax(x)
    if i < half_num_arround:
        return x[:half_num_arround*2]
    elif i+half_num_arround > 512:
        return x[-(half_num_arround*2):]

    return x[i - half_num_arround:i + half_num_arround]

