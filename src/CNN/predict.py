

df_test = df_train[n_train:].reset_index(drop=True)
print(df_test.shape)


test_dataset = OpticsDataset(
                    df=df_test,
              )
    
test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=256,
            num_workers=0, 
            pin_memory=True
        )

EXP_ID = 'exp1'
device = 'cuda'

model_path = [
        f'{EXP_ID}_fold0.pth',
        f'{EXP_ID}_fold1.pth',
        f'{EXP_ID}_fold2.pth',
        f'{EXP_ID}_fold3.pth',
        f'{EXP_ID}_fold4.pth',
]

if 1:
    model_path = [f'{EXP_ID}_fold0.pth']

print(model_path)
    
models = []
for p in tqdm(model_path):
    model = SimpleModel(4, 512)
    model.to(device)
    model.load_state_dict(torch.load(p))
    model.eval()
    models.append(model)
    
    
model = models[0]


test_preds = test_fn(test_loader, model, 'cuda')
test_preds = np.concatenate(test_preds, 0)
print(test_preds.shape)


file_path = '../input/atma5-data/'

sub = pd.read_csv(os.path.join(file_path, 'atmaCup5__sample_submission.csv'))
sub.target = test_preds

print(sub.shape)
print(sub.target.value_counts())
print(sub.target.describe())

sub.to_csv('late_sub.csv', index=False)

sub.head()
