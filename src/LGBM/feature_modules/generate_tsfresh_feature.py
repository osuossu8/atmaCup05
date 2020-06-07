# !pip install tsfresh > /dev/null
from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import settings


X = extract_features(spectrum_df,
                     column_id="spectrum_filename",
                     column_sort="wl",
                     n_jobs=8,
                     default_fc_parameters=settings.EfficientFCParameters()
                     )

# 0 だけとかの feature を落とす
use_cols = []
for c in X.columns:
    if max(X[c]) == min(X[c]):
        continue
    use_cols.append(c)

X = X[use_cols].reset_index()

to_pickle('tsfresh_features_EfficientFCParameters.pkl', X)
tsfresh_feature = unpickle('tsfresh_features_EfficientFCParameters.pkl')

print(tsfresh_feature.shape)
tsfresh_feature.head()
