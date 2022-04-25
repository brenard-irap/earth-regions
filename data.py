import speasy as spz
import pandas as pd
import numpy as np
import datetime
import more_itertools


def get_df_from_speasy(start, stop):
    data_amda_config = [
        {
            "id": "mms1_b_gse",
            "columns": ["bx", "by", "bz"]
        },
        {
            "id": "mms1_b_tot",
            "columns": ["btot"]
        },
        {
            "id": "mms1_dis_ni",
            "columns": ["n"]
        },
        {
            "id": "mms1_dis_vgse",
            "columns": ["vx", "vy", "vz"]
        },
        {
            "id": "mms1_dis_tpara",
            "columns": ["T_para"]
        },
        {
            "id": "mms1_dis_tperp",
            "columns": ["T_perp"]
        }
    ]

    df_array = []
    for data in data_amda_config:
        res = spz.get_data('amda/{}'.format(data["id"]), start, stop)
        columns_mapper = {}
        if len(data["columns"]) == 1:
            columns_mapper[data["id"]] = data["columns"][0]
        elif len(data["columns"]) > 1:
            for col in range(len(data["columns"])):
                columns_mapper["{}[{}]".format(data["id"], col)] = data["columns"][col]
        df = res.to_dataframe(datetime_index=True)
        df = df.rename(columns=columns_mapper)
        df_array.append(df)

    return df_array


def merge_df(df_array, start, sampling):
    df_merged = pd.DataFrame()
    for df in df_array:
        df = df.resample('{}S'.format(sampling), origin=datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")).mean()
        if df_merged.empty:
            df_merged = df
        else:
            df_merged = pd.merge(df_merged, df, left_index=True, right_index=True)
    return df_merged


def inject_additionnal_features(df_merged):
    df_merged.insert(df_merged.columns.get_loc('vz') + 1, "vtot",
                     (df_merged["vx"] ** 2 + df_merged["vy"] ** 2 + df_merged["vz"] ** 2) ** (1 / 2))
    df_merged.insert(df_merged.columns.get_loc('T_perp') + 1, "T_tot", (df_merged['T_para'] + df_merged['T_perp']) / 2.)


def prepare_data(df_merged, block_size):
    mean_values = {
        "bx": 2.088362511781322,
        "by": 2.3189843400918217,
        "bz": 10.148729235092244,
        "btot": 20.494615970470377,
        "n": 8.372533778049707,
        "vx": -126.79884925763078,
        "vy": 26.595425024996928,
        "vz": 9.67785452048571,
        "vtot": 174.84539947528972,
        "T_para": 1540.9134603866269,
        "T_perp": 1706.1844820725112,
        "T_tot": 1623.548971275055
    }

    std_values = {
        "bx": 11.761082436068994,
        "by": 10.926197414288536,
        "bz": 18.806352592962885,
        "btot": 17.438179422155088,
        "n": 14.850501418501615,
        "vx": 176.14052230960277,
        "vy": 71.51236378276397,
        "vz": 41.53659888041524,
        "vtot": 155.47605210032796,
        "T_para": 1916.4463569295974,
        "T_perp": 2174.352772259333,
        "T_tot": 2031.4815960107615
    }

    blocks = list(more_itertools.chunked(df_merged.values, block_size))
    blocks = [np.array(x) for x in blocks]

    x_input = np.array(blocks[:-1])
    timestamps = list(more_itertools.chunked(df_merged.index, block_size))
    timestamps = np.array(timestamps[:-1])

    index = pd.DatetimeIndex(timestamps[:, 0])

    output = []
    for feature in df_merged.columns:
        col_index = list(df_merged.columns).index(feature)
        mean_value = mean_values[feature]
        std_value = std_values[feature]
        standardize_output = (x_input[:, :, col_index] - mean_value) / (std_value)
        output.append(standardize_output)

    x_input = np.stack(output, axis=2)

    return index, x_input
