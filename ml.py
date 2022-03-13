# region imports

import warnings
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GroupKFold
import catboost

import stats
import tools


# endregion


# region helper functions
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def drop_pid_small_data(df, low_boundary=60):
    """
    The function drops pids with smaller amount of samples than low_boundary
    The default low boundary value is 60 which corresponds to 2 months
    """
    df_samples = tools.get_samples_per_group(df, 'pid')
    drop_pid_list = list(df_samples.loc[df_samples['samples_num'] < low_boundary]['pid'])

    return df[~df['pid'].isin(drop_pid_list)]


def drop_pid_small_ema_var(df, low_boundary=0.0):
    """
    The function drops pids with smaller than (or equal to) low_boundary of total EMA variance
    """
    df_ema_variance = stats.get_total_ema_variation_by_each_symptom(df)
    drop_pid_list = list(df_ema_variance.loc[df_ema_variance['var_sum'] <= low_boundary]['pid'])
    return df[~df['pid'].isin(drop_pid_list)]


def filter_out_participants(df, low_boundary_samples=60, low_boundary_ema_var=6.99):
    df = df[df['pid'].astype(str).str.startswith(f'{tools.CAMPAIGN_5}0')]
    print(f'Initial number of participants: {df["pid"].nunique()}')

    df = df[~df['pid'].isin(tools.DROP_PARTICIPANT_LIST)]

    # remove participants with less than 60 data samples (2 months of data)
    df = drop_pid_small_data(df, low_boundary=low_boundary_samples)
    print(
        f'Number of participants after removing pids with EMA quantity less than {low_boundary_samples}: {df["pid"].nunique()}')

    # remove participants with EMA variance less than boundary
    df = drop_pid_small_ema_var(df, low_boundary=low_boundary_ema_var)
    print(
        f'Number of participants after removing pids with EMA variance less than or equal to {low_boundary_ema_var}: {df["pid"].nunique()}')

    return df


# endregion

# region ml general functions
def perf_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    metrics = []
    y_pred_cls = np.rint(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = f1_score(y_true, y_pred_cls)
        prec_score = precision_score(y_true, y_pred_cls)
        rec_score = recall_score(y_true, y_pred_cls)
        # spec_score = specificity_score(y_true, y_pred_cls)

    metrics.append(dict(F1=f1, PRECISION=prec_score, RECALL=rec_score))
    return reduce(lambda a, b: dict(a, **b), metrics)


def group_inner_split(X_train, y_train, pids):
    inner_splitter = GroupKFold(n_splits=5)
    for dev_index, val_index in inner_splitter.split(X_train, y_train, groups=pids):
        return dev_index, val_index


def LOSOCatBoost(X, y, feature_names, pids, gt):
    splitter = GroupKFold(n_splits=len(np.unique(pids)))
    results = []
    for i, (train_index, test_index) in enumerate(splitter.split(X, y, groups=pids)):
        results.append(run_loso_trial(X, y, train_index, test_index, feature_names, gt))
    results = pd.DataFrame(results)
    results.insert(0, 'CV_TYPE', 'LOSO', allow_duplicates=True)
    results.insert(0, 'GT', gt, allow_duplicates=True)

    return results


def run_loso_trial(X, y, train_index, test_index, feature_names, gt):
    pids = X['pid']
    pid = X['pid'].iloc[test_index].unique()[0]
    X = X[feature_names]

    cb_clf = catboost.CatBoostClassifier(random_seed=tools.RANDOM_SEED, depth=10, learning_rate=0.05, iterations=120,
                                         l2_leaf_reg=3)
    dev_index, val_index = group_inner_split(X.iloc[train_index][feature_names], y.iloc[train_index],
                                             pids.iloc[train_index])

    d_dev = catboost.Pool(
        data=X.iloc[train_index].iloc[dev_index][feature_names],
        label=y.iloc[train_index].iloc[dev_index],
        feature_names=feature_names
    )

    d_val = catboost.Pool(
        data=X.iloc[train_index].iloc[val_index][feature_names],
        label=y.iloc[train_index].iloc[val_index],
        feature_names=feature_names
    )

    cb_clf.fit(X=d_dev,
               use_best_model=True,
               eval_set=d_val,
               verbose_eval=False,
               early_stopping_rounds=35,
               )

    prob = cb_clf.predict_proba(X.iloc[test_index][feature_names])[:, 1]
    metrics = perf_metrics(y.iloc[test_index], prob)
    res = metrics
    res.update({'pid': pid})

    # feat_importances = pd.Series(cb_clf.feature_importances_, index=X.columns)

    return res


def run_loso_classification(args):
    df = args[0]
    gt = args[1]

    print(f'\nClasses distribution ({gt}): \n{df[gt].value_counts()}')
    df[gt].value_counts().plot.bar(title=f'Classes distribution: {gt}')

    M_features = df.columns.str.contains('#')
    feature_names = list(df.columns[M_features])
    feature_names.append('pid')
    X = df[feature_names]

    # X = preprocess.normalize_dataframe(X, feature_names)
    # X = preprocess.binnarize_dataframe(X, feature_names)

    y = df[gt]
    pids = df['pid']
    feature_names.remove('pid')

    return [LOSOCatBoost(X, y, feature_names, pids, gt), gt]
# endregion
