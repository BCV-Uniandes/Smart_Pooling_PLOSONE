import  h2o
from h2o.automl import H2OAutoML


def prepare_data_train(train, val=None):
    x = train.columns
    y = x[-1]
    x.remove(y)
    train[y] = train[y]
    if val is not None:
        val[y] = val[y]
        return train, x, y, val
    return train, x, y

def train_models(train, val, experiment_name):
    
    train, x, y, val = prepare_data_train(train, val)

    models = H2OAutoML(max_runtime_secs=300,
                       seed=1,
                       exclude_algos=['DeepLearning'],
                       project_name=experiment_name,
                       nfolds=0,
                       sort_metric='MSE')

    models.train(x=x, y=y, training_frame=train, validation_frame=val, leaderboard_frame=val)

    return models


def retrain_best_model(complete_train_df, best_model, save_model=True, save_path='experiment_retrained'):
    # TODO: add for GBM training w/ quantile
    # organize data for h2o
    train, x, y = prepare_data_train(complete_train_df)
    best_model.train(x=x, y=y,
                     training_frame=train)
    if save_model:
        h2o.save_model(model=best_model, path=save_path, force=True)
    return best_model