import sklearn
import matplotlib.pyplot as plt


fig, ax = plt.subplots(2, 6, figsize=(20, 6))
for i in tqdm(range(6)):
    # 負例の波形データの可視化
    wv1 = df_train.iloc[list(df_train[df_train.target == 0].index)]['peak_around_wave'].reset_index(drop=True)[i]
    ax[i//6][i%6].plot(ss_scaler.fit_transform(wv1.reshape(-1, 1)), color='b')

for i in tqdm(range(6, 12)):
    # 正例の波形データの可視化
    wv1 = df_train.iloc[list(df_train[df_train.target == 1].index)]['peak_around_wave'].reset_index(drop=True)[i]
    ax[i//6][i%6].plot(ss_scaler.fit_transform(wv1.reshape(-1, 1)), color='r')
