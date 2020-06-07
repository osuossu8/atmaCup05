import sklearn


def extract_cnn_feature(w1, w2, w3):
    w1 = ss_scaler.fit_transform(w1.reshape(-1, 1))
    w2 = ss_scaler.fit_transform(w2.reshape(-1, 1))
    w3 = ss_scaler.fit_transform(w3.reshape(-1, 1))  
    w0 = np.vstack([w2, w3])
    w = np.vstack([w0, w1]).reshape(-1, 2, 128) 
    # oof_preds = model(torch.tensor(w, dtype=torch.float32), torch.tensor(0)).detach().numpy() 
    oof_preds = model(torch.tensor(w, dtype=torch.float32)).detach().numpy() 
    return oof_preds[0]


def map_func(row):
    w1 = row['peak_around_light_intensity']
    w2 = row['log_fftn_peak_around_light_intensity']
    w3 = row['log_fftn_peak_around_dft_polar_light_intensity'] 
    return extract_cnn_feature(w1, w2, w3)


ss_scaler = sklearn.preprocessing.StandardScaler()
df_train['log_fftn_peak_around_light_intensity'] = df_train['log_fftn_peak_around_light_intensity'].apply(lambda x: paddingV2(x, 64))
df_train['log_fftn_peak_around_dft_polar_light_intensity'] = df_train['log_fftn_peak_around_dft_polar_light_intensity'].apply(lambda x: paddingV2(x, 64))

oof_features = []
for i in range(5):
    use_idx = splits[i][1]
    weight = torch.load(f'/content/drive/My Drive/offline_competition/atma05/results/exp1_fold{i}.pth',
                        map_location=torch.device('cpu'))
    model = Optics1dCNNModelV3()
    model.load_state_dict(weight)
    print(f'weight {i} loaded')
    model = nn.Sequential(*list(model.children())[:-3])    

    fold_df = df_train.loc[use_idx]
    oof_features.append(fold_df.progress_apply(map_func, axis=1)) 

weight = torch.load(f'/content/drive/My Drive/offline_competition/atma05/results/exp1_fold3.pth',
                    map_location=torch.device('cpu'))
model = Optics1dCNNModelV3()
model.load_state_dict(weight)    
print(f'weight {i} loaded')
model = nn.Sequential(*list(model.children())[:-3])

oof_features.append(df_train.loc[n_train:].progress_apply(map_func, axis=1)) 

df_train['cnn_attn'] = np.array([f[0] for f in np.hstack(oof_features)])
attn_df = pd.DataFrame(np.vstack(np.hstack(oof_features))).add_prefix('attn_')
attn_cols = list(attn_df.columns)

df_train = pd.concat([df_train, attn_df], 1)
attn_df.head()
