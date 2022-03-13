# region imports
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# endregion

# region constants
FOLDS_NUM = 2
RANDOM_SEED = 44
CAMPAIGN_4 = 4
CAMPAIGN_5 = 5

PREP_DATA_PATH = 'data/FS_campaign_4_5-WIN_4_preprocessed_Jan.csv'
LOSO_RESULTS_BIN_PATH = 'results/loso/bin'
LOSO_RESULTS_BIN_NOSUB_PATH = 'results/loso/bin_nosub'
TOOLS_PATH = 'tools'
STATS_PATH = 'stats'

SYMPTOM_BIN_COLUMN_LIST = ["lack_of_interest_bin",
                           "depressed_feeling_bin",
                           "sleep_trouble_bin",
                           "fatigue_bin",
                           "poor_appetite_bin",
                           "negative_self_image_bin",
                           "difficulty_focusing_bin",
                           "bad_physchomotor_activity_bin",
                           "suicide_thoughts_bin"
                           ]

SYMPTOM_BIN_NOSUB_COLUMN_LIST = ["lack_of_interest_bin_nosub",
                                 "depressed_feeling_bin_nosub",
                                 "sleep_trouble_bin_nosub",
                                 "fatigue_bin_nosub",
                                 "poor_appetite_bin_nosub",
                                 "negative_self_image_bin_nosub",
                                 "difficulty_focusing_bin_nosub",
                                 "bad_physchomotor_activity_bin_nosub",
                                 "suicide_thoughts_bin_nosub"
                                 ]

SYMPTOM_DIF_NOSUB_COLUMN_LIST = ["lack_of_interest_dif_nosub",
                                 "depressed_feeling_dif_nosub",
                                 "sleep_trouble_dif_nosub",
                                 "fatigue_dif_nosub",
                                 "poor_appetite_dif_nosub",
                                 "negative_self_image_dif_nosub",
                                 "difficulty_focusing_dif_nosub",
                                 "bad_physchomotor_activity_dif_nosub",
                                 "suicide_thoughts_dif_nosub"
                                 ]

SYMPTOM_COLUMN_LIST = ["lack_of_interest",
                       "depressed_feeling",
                       "sleep_trouble",
                       "fatigue",
                       "poor_appetite",
                       "negative_self_image",
                       "difficulty_focusing",
                       "bad_physchomotor_activity",
                       "suicide_thoughts"
                       ]

SYMPTOM_COLUMN_NAME_LIST = ["Diminished interest",
                            "Depressed mood",
                            "Sleep troubles",
                            "Fatigue",
                            "Appetite problems",
                            "Worthlessness",
                            "Inability to concentrate",
                            "Psychomotor activity",
                            "Suicidal thoughts"
                            ]

DROP_PARTICIPANT_LIST = [4082, 4084, 4096,  # Android version problem
                         5072, 50196, 50189, 50417, 50370, 50215,  # 0 variation in total phq score
                         50179, 50230, 50538, 50628, 50630,  # have 1 EMA only
                         50702, 50710, 50769, 50514, 50286, 50477, 50771, 50765, 50184, 50365, 50389, 50759, 50559,
                         50513, 50343, 50692, 50742, 50741, 50738, 50691, 50700, 50577, 50733]  # have less than 10 EMAs


# endregion

# region general function definitions
def create_dir_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def subtract_lists(l1, l2):
    return [x for x in l1 if x not in l2]


# endregion

# region data related function definitions
def split_file_by_campaigns(filename):
    df = pd.read_csv(filename)

    for campaign_id in tqdm([CAMPAIGN_4, CAMPAIGN_5]):
        df_cmp = df[df['pid'].astype(str).str.startswith(f'{campaign_id}0')]

        filename_out = f"{filename.split('.')[0]}_cmp{campaign_id}.csv"
        df_cmp.to_csv(filename_out, index=False)


def get_samples_per_group(df, group):
    df_out = pd.DataFrame()
    df_out['samples_num'] = df.groupby(group).size()
    df_out.sort_values(by=['samples_num'], inplace=True)
    df_out = df_out.rename_axis('pid').reset_index()

    create_dir_if_not_exists(TOOLS_PATH)
    df_out.to_csv(f'{TOOLS_PATH}/samples_per_{group}.csv')
    return df_out


def remove_subjectivity_from_gt(filename):
    df = pd.read_csv(filename)
    frames = []
    pid_group_list = df.groupby(by=['pid'])
    for name, pid_group in pid_group_list:
        for symptom in SYMPTOM_COLUMN_LIST:
            symptom_mean = pid_group[symptom].mean()
            pid_group[f'{symptom}_dif_nosub'] = pid_group[symptom] - symptom_mean
        frames.append(pid_group)
    print(f'Before: {df.shape}')
    df = pd.concat(frames)
    print(f'After: {df.shape}')
    df.to_csv(filename, index=False)

    # endregion


def get_ground_truth_stats(filename):
    df = pd.read_csv(filename)

    df_out = pd.DataFrame()
    no_symptom_count = []
    symptom_count = []

    for symptom in SYMPTOM_BIN_NOSUB_COLUMN_LIST:
        no_symptom_count.append(df[symptom].value_counts().get(0))
        symptom_count.append(df[symptom].value_counts().get(1))

    df_out['symptom'] = SYMPTOM_BIN_COLUMN_LIST
    df_out['no_count'] = no_symptom_count
    df_out['yes_count'] = symptom_count

    create_dir_if_not_exists(STATS_PATH)
    df_out.to_csv('stats/symptom_bin_nosub_dist.csv', index=False)
