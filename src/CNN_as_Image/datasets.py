

class ImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, training=True):
        self.df = df
        self.training = training
        
    def __len__(self):
        return len(self.df)
    
    
    def transform(self, image, SIZE):
        data_transforms = albumentations.Compose([
                                    albumentations.CenterCrop(int(SIZE[1] * 0.75), int(SIZE[0] * 0.75), p=1),
                                    albumentations.Resize(224, 224, p=1),
                          ])        
        return data_transforms(image=image)['image'] 
    
    
    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        
        file_name = row.spectrum_filename.split('.')[0]
        
        if self.training:
            image = Image.open(f'../input/atma5-image/{file_name}.jpg')
        else:
            image = Image.open(f'../input/atma5-image-test/{file_name}.jpg')
        
        SIZE = image.size       
        image = np.array(image)

        image = self.transform(image, SIZE)        
        tensor = image_to_tensor(image)
        
        return {
            'image': tensor,
            'spectrum_filename': row.spectrum_filename,
        }


class ImageDatasetV2(torch.utils.data.Dataset):
    
    def __init__(self, df, y=None, training=True):
        self.df = df
        self.y = y
        self.training = training
        
    def __len__(self):
        return len(self.df)
    
    
    def transform(self, image, SIZE):
        if self.training:
            data_transforms = albumentations.Compose([
                                    albumentations.CenterCrop(int(SIZE[1] * 0.75), int(SIZE[0] * 0.75), p=1),
                                    # albumentations.Cutout(p=0.2),
                                    # albumentations.Resize(224, 224, p=1),
                                    albumentations.Resize(728, 728, p=1),
                                    # albumentations.Normalize(p=1.0),
                                    ToTensorV2()
                              ])
        else:
            data_transforms = albumentations.Compose([
                    albumentations.CenterCrop(int(SIZE[1] * 0.75), int(SIZE[0] * 0.75), p=1),
                    # albumentations.Resize(224, 224, p=1),
                    albumentations.Resize(728, 728, p=1),
                    # albumentations.Normalize(p=1.0),
                    ToTensorV2()
                ])            
        return data_transforms(image=image)['image'] 
    
    
    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        
        file_name = row.spectrum_filename.split('.')[0]
        
        if self.training:
            image = Image.open(f'../input/atma5-image/{file_name}.jpg')
        else:
            image = Image.open(f'../input/atma5-image-test/{file_name}.jpg')
        
        SIZE = image.size       
        image = np.array(image)

        image_tensor = self.transform(image, SIZE)  
        
        if self.y is not None:      
        
            return {
                'images': image_tensor,
                'targets': torch.tensor(y[idx], dtype=torch.float32)
            }
        
        return {
            'images': image_tensor
        }
