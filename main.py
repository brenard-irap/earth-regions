from data import get_df_from_speasy, merge_df, prepare_data, inject_additionnal_features
from model import run_prediction
from export import save_as_amda_catalog
import argparse
import os


def main():
    block_size = 40
    sampling = 4.5

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--start", type=str, default='2019-11-09T00:00:00')
    parser.add_argument("--stop", type=str, default='2019-11-10T00:00:00')

    args = parser.parse_args()

    destination_folder_path = args.output_dir
    start = args.start
    stop = args.stop

    df_array = get_df_from_speasy(start, stop)
    df_merged = merge_df(df_array, start, sampling)
    inject_additionnal_features(df_merged)
    index, x_input = prepare_data(df_merged, block_size)

    y_classes = run_prediction(x_input, index)

    save_as_amda_catalog(y_classes, block_size*sampling, os.path.join(destination_folder_path, 'output_classes.cat'))


if __name__ == '__main__':
    main()
