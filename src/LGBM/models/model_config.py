

def pr_auc(preds, data):
    y_true = data.get_label()
    score = average_precision_score(y_true, preds)
    return "pr_auc", score, True


lgb_config = {
    "model": {
        "name": "lightgbm",
        "model_params": {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": 'None', # "binary",
            "learning_rate": 0.1,
            "num_leaves": 12,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "min_data_in_leaf": 30,
            "seed": 71,
            "bagging_seed": 71,
            "feature_fraction_seed": 71,
            "drop_seed": 71,
            "verbose": -1
        },
        "train_params": {
            "num_boost_round": 5000,
            "early_stopping_rounds": 200,
            "verbose_eval": 500,
            'feval': pr_auc,
        }
    },
}


cat_config = {
    "model": {
        "model_params": {
            'loss_function': 'Logloss',
            'learning_rate': 0.1,
            'iterations': 800,
            'early_stopping_rounds': 150,
        },
    }
}
