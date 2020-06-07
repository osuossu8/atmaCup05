

class OpticsDataset:
    def __init__(self, df, indices=None):
        if indices is not None:
            self.target = df.iloc[indices].target.values
            self.spectrum_filename = df.iloc[indices].spectrum_filename.values
        else:
            self.target = df.target.values
            self.spectrum_filename = df.spectrum_filename.values 

        self.df = df
        self.mm_scaler = sklearn.preprocessing.MinMaxScaler()
        self.ss_scaler = sklearn.preprocessing.StandardScaler()

    def __len__(self):
        return len(self.target)

    def process_data(self, target, spectrum_filename):

        spec = self.df[self.df.spectrum_filename==spectrum_filename]['light_intensity'].iloc[0] 
        
        light_intensity = np.stack([
            spec.reshape(1, -1),  # 生のスペクトル
            scipy.signal.savgol_filter(spec.reshape(1, -1), 5, 2, deriv=0, axis=1),  # なめらかにしただけ   
            scipy.signal.savgol_filter(spec.reshape(1, -1), 5, 2, deriv=1, axis=1),  # 1次微分
            scipy.signal.savgol_filter(spec.reshape(1, -1), 5, 2, deriv=2, axis=1),  # 2次微分
        ], axis=1)
        
        targets = [1] if target == 1 else [0]

        return {
            'light_intensity' : light_intensity,
            'targets' : targets,
        }

    def __getitem__(self, item):
        data = self.process_data(
            self.target[item], 
            self.spectrum_filename[item],
        )

        return {
            'light_intensity': torch.tensor(data["light_intensity"], dtype=torch.float32),
            'targets': torch.tensor(data["targets"], dtype=torch.float32),
        }
