

class AtmaOpticsImageModel(nn.Module):
    
    def __init__(self, num_classes=1):
        super().__init__()
        
        self.backbone = torchvision.models.resnet34(pretrained=True)
        
        in_features = self.backbone.fc.in_features

        self.dropout = nn.Dropout(0.25)
        self.logit = nn.Linear(in_features, num_classes)
        
    def forward(self, x):

        batch_size, C, H, W = x.shape
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = torch.nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = self.dropout(x)
        
        x = self.logit(x)
        return x
