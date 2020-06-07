

from sklearn.manifold import TSNE

spec_meta_df['light_intensity'] = spec_meta_df['light_intensity'].apply(lambda x: padding(x))

clf = TSNE(n_components=2)
z = clf.fit_transform(np.stack(spec_meta_df['light_intensity'].values))
projected_df = pd.DataFrame(z, columns=['project_0', 'project_1'])

print(projected_df.shape)
projected_df.head()
