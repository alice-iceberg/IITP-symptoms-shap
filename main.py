import concurrent.futures

import pandas as pd
from tqdm import tqdm

import ml
import tools


def machine_learning():
    df = pd.read_csv(tools.PREP_DATA_PATH)
    df = ml.filter_out_participants(df)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(ml.run_loso_classification, [df, symptom]) for symptom in
                   tqdm(tools.SYMPTOM_BIN_NOSUB_COLUMN_LIST)]

    for f in concurrent.futures.as_completed(results):
        args = f.result()
        df_out = args[0]
        symptom = args[1]

        tools.create_dir_if_not_exists(tools.LOSO_RESULTS_BIN_NOSUB_PATH)
        df_out.to_csv(f'{tools.LOSO_RESULTS_BIN_NOSUB_PATH}/{symptom}_loso_bin_nosub.csv', index=False)


def main():
    machine_learning()


if __name__ == '__main__':
    main()
