# region imports
from functools import reduce

import pandas as pd

import tools


# endregion


# region variance function definitions
def get_feature_variation_per_participant(df, feature):
    df_std = pd.DataFrame()
    df_std[f'{feature}_var'] = df.groupby(['pid'])[feature].std()
    df_std.sort_values(by=[f'{feature}_var'], inplace=True, ascending=False)

    return df_std


def get_total_ema_variation_by_each_symptom(df):  # todo recheck the columns (nosub/bin)
    df_symptom_var_list = []
    for symptom in tools.SYMPTOM_DIF_NOSUB_COLUMN_LIST:  # todo recheck the column list
        df_symptom_var = get_feature_variation_per_participant(df, symptom)
        df_symptom_var_list.append(df_symptom_var)

    df_symptoms_merged = reduce(lambda left, right: pd.merge(left, right, on=['pid'],
                                                             how='outer'), df_symptom_var_list)
    df_symptoms_merged['var_sum'] = df_symptoms_merged["negative_self_image_dif_nosub_var"] + df_symptoms_merged[
        # todo recheck the column names
        "sleep_trouble_dif_nosub_var"] + \
                                    df_symptoms_merged["depressed_feeling_dif_nosub_var"] + df_symptoms_merged[
                                        "suicide_thoughts_dif_nosub_var"] + \
                                    df_symptoms_merged["difficulty_focusing_dif_nosub_var"] + df_symptoms_merged[
                                        "poor_appetite_dif_nosub_var"] + \
                                    df_symptoms_merged["lack_of_interest_dif_nosub_var"] + df_symptoms_merged[
                                        "fatigue_dif_nosub_var"] + \
                                    df_symptoms_merged["bad_physchomotor_activity_dif_nosub_var"]

    df_symptoms_merged.sort_values(by=["var_sum"], inplace=True, ascending=False)
    df_symptoms_merged = df_symptoms_merged.rename_axis('pid').reset_index()
    df_symptoms_var_desc = df_symptoms_merged.describe()

    tools.create_dir_if_not_exists(tools.TOOLS_PATH)
    df_symptoms_var_desc.to_csv(f'{tools.TOOLS_PATH}/ema_var_nosub_dif_stats.csv')  # todo recheck the filename

    # df_symptoms_merged['var_sum'].plot(kind='hist', bins=50, title='EMA total variance distribution')

    return df_symptoms_merged
# endregion
