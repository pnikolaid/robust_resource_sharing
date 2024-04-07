from parameters import slices, zones, freqs, start_times, end_times, days, test_scenario

import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pickle

pd.set_option('display.max_columns', None)

# Raw file format: [tiemstamp, sfn, subframe, rnti, direction, mcs_idx, nof_prb] (direction=1 means downlink)
for data_index in range(slices):

    data_index = 3  # collect data for each slice one-by-one

    zone = zones[data_index]
    freq = freqs[data_index]

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Processing slice with data {zone}-{freq}")

    file_string = zone + '-' + freq + '-raw-df-ms.parquet'
    pq_file = pq.ParquetFile(file_string)

    first_batch = True
    for batch in pq_file.iter_batches():
        df = batch.to_pandas()
        df.dropna(inplace=True)

        first_timedate = datetime.fromtimestamp(df['timestamp'][df.index[1]], ZoneInfo(key='Europe/Madrid'))
        last_timedate = datetime.fromtimestamp(df['timestamp'][df.index[-1]], ZoneInfo(key='Europe/Madrid'))
        print(f"Batch first timestamp: {first_timedate}")
        print(f"Batch last timestamp: {last_timedate}")

        within_observation_period = False
        current_day = -1
        for i in range(days):

            # Compute endpoints of day observation period
            observation_day_start = start_times[data_index] + timedelta(i)
            observation_day_end = end_times[data_index] + timedelta(i)

            if first_timedate >= observation_day_start and last_timedate <= observation_day_end:
                current_day = i + 1
                within_observation_period = True
                break

        # log the batch if within observation period
        if within_observation_period:
            batch_list = df.to_numpy().tolist()
            print(f"Batch within Day {current_day}/{days}! Appended to pickle file!")
            if first_batch:
                with open(f"ts{test_scenario}_" + zone + freq + '-alldata' + '.pkl', 'wb') as f:
                    pickle.dump(batch_list, f)
                    first_batch = False
            else:
                with open(f"ts{test_scenario}_" + zone + freq + '-alldata' + '.pkl', 'ab') as f:
                    pickle.dump(batch_list, f)

        # if the batch starts after the last time of the last day, the search needs to terminate
        elif first_timedate > end_times[data_index] + timedelta(days):
            print("Batch starts after the observation period ends -> stopping search!")
            break
    break    # collect data for each slice one-by-one
