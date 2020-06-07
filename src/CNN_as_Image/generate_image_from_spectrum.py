from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


spec_meta_df = unpickle('../input/atma5-data/spectrum_raw_train.pkl')

for i, fname in tqdm(enumerate(spec_meta_df.spectrum_filename.unique())):
    intensity = spec_meta_df[spec_meta_df['spectrum_filename']==fname]['wave'].values[0]

    fig = plt.figure()
    plt.plot(intensity)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.FixedLocator([])) # y 目盛りを消す
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.FixedLocator([])) # x 目盛りを消す

    save_name = fname.split('.')[0]

    plt.savefig(f'{save_name}.jpg')
