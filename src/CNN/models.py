

class SimpleModel(nn.Module):
    """
    private 6th place amane`s model
    """
    def __init__(self, in_channels, n_features, hidden_channels=64, out_dim=1):
        super().__init__()
        self.filters = [3, 5, 7, 21, 51, 101]
        for filter_size in self.filters:
            setattr(
                self,
                f"seq{filter_size}", 
                nn.Sequential(
                    Conv1dBlock(in_channels, hidden_channels, filter_size, dropout=0.1),
                    Conv1dBlock(hidden_channels, hidden_channels, filter_size, dropout=0.1),
                ),
            )
        self.cont = nn.Sequential(
            nn.Linear(n_features, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.last_linear = nn.Sequential(
            # nn.Linear(hidden_channels*(len(self.filters)+1), hidden_channels),
            nn.Linear(hidden_channels*(len(self.filters)), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_dim),
            # nn.Sigmoid()
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    # def forward(self, in_seq, in_cont):
    def forward(self, in_seq):
        outs = []
        for filter_size in self.filters:
            out = getattr(self, f"seq{filter_size}")(in_seq)
            out, _ = torch.max(out, -1)
            outs.append(out)

        # outs.append(self.cont(in_cont))
        out = torch.cat(outs, axis=1)
        out = self.last_linear(out)
        return out # .flatten()


class Optics1dCNNModelV8(nn.Module):
    def __init__(self):
        super(Optics1dCNNModelV8, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            CSE1D(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=7, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            CSE1D(16)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=7, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            CSE1D(8)
        ) 

        self.drop_out = nn.Dropout(0.2)
        
        self.regressor = nn.Sequential(
            nn.Linear(4000, 512),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def forward(self, light_intensity):
        h_conv = self.conv1(light_intensity)
        h_conv = self.conv2(h_conv)
        h_conv = self.conv3(h_conv)
        h_conv = h_conv.view(h_conv.size(0), -1)
        output = self.drop_out(h_conv)
        output = self.regressor(output)
        return output


class Optics1dCNNModelV3(nn.Module):
    def __init__(self):
        super(Optics1dCNNModelV3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),)

        self.conv_attention = Attention(126, 128)
        self.drop_out = nn.Dropout(0.2)
        self.linear = nn.Linear(len(num_cols), 32)
        
        self.regressor = nn.Sequential(
            nn.Linear(126, 64),
            nn.Dropout(),
            nn.Linear(64, 1)
        )

    def forward(self, light_intensity, features):
        h_conv = self.conv1(light_intensity)
        h_attn = self.conv_attention(h_conv)
        output = self.drop_out(h_attn)
        output = self.regressor(output)
        return output
