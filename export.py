import numpy as np
from datetime import datetime


def save_as_amda_catalog(y_classes, gap_size, catalog_path):
    time_format = "%Y-%m-%dT%H:%M:%S"
    intervals = []
    crt_val = None
    crt_start = None
    prev_time = None
    for index, row in y_classes.iterrows():
        if crt_val is None:
            crt_val = y_classes.at[index, 'classes']
            crt_start = y_classes.at[index, 'Time']
        elif (y_classes.at[index, 'classes'] != crt_val) or \
                (prev_time and ((y_classes.at[index, 'Time'] - prev_time).total_seconds() > gap_size)):
            if (y_classes.at[index, 'Time'] - prev_time).total_seconds() > gap_size:
                row = [crt_start.strftime(time_format), prev_time.strftime(time_format), crt_val]
            else:
                row = [crt_start.strftime(time_format), y_classes.at[index, 'Time'].strftime(time_format), crt_val]
            intervals.append(row)
            crt_val = y_classes.at[index, 'classes']
            crt_start = y_classes.at[index, 'Time']
        prev_time = y_classes.at[index, 'Time']

    if crt_start and (crt_start != y_classes.at[y_classes.index.values[-1], 'Time']):
        row = [crt_start.strftime(time_format), y_classes.at[y_classes.index.values[-1], 'Time'].strftime(time_format), crt_val]
        intervals.append(row)

    catalog = np.array(intervals)
    np.savetxt(catalog_path, catalog, delimiter=" ", fmt='%s', header="Parameter 1: id:param_0; name:classes; size:1; type:string; unit:; description:0 : SW [241,196,15] - 1 : FS [26, 188, 156] - 2 : BS [39, 176, 96] - 3 : MSH [41, 128, 185] - 4 : MP [155, 89, 182] - 5 : BL [192, 57, 43] - 6 : MSP [211, 84, 0] - 7 : PS [127, 140, 141] - 8 : PSBL [44, 62, 80] - 9 : LOBE [0, 0, 255]; ucd:; utype:;")

