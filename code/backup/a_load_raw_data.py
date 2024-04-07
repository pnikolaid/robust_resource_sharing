from parameters import slices, zones, freqs, start_times, end_times, days, test_scenario

import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from bisect import bisect_left
import pickle

pd.set_option('display.max_columns', None)

# Raw file format: [tiemstamp, sfn, subframe, rnti, direction, mcs_idx, nof_prb] (direction=1 means downlink)
for data_index in range(slices):

    zone = zones[data_index]
    freq = freqs[data_index]
    first_start_time = start_times[data_index]
    first_end_time = end_times[data_index]

    final_chunks = 0
    final_list = []
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Processing slice with data {zone}-{freq}")
    # Scan slice traffic for multiple
    for i in range(days):
        start_time = first_start_time + timedelta(days=i)
        end_time = first_end_time + timedelta(days=i)

        print("-------------------------------------------------------------------------------------------------------")
        print(f"Day {i+1}/{days}")
        print(f"Target period: {start_time} to {end_time}")
        # Specify the input Parquet file and output CSV file
        parquet_file = zone + '-' + freq + '-raw-df-ms.parquet'

        pq_file = pq.ParquetFile(parquet_file)

        # Define data collection period
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()
        observation_list = []
        chunks = 0
        # TODO: continue searching where it left off
        for batch in pq_file.iter_batches():
            df = batch.to_pandas()
            df.dropna(inplace=True)
            first_timestamp = datetime.fromtimestamp(df['timestamp'][df.index[1]], ZoneInfo(key='Europe/Madrid'))
            last_timestamp = datetime.fromtimestamp(df['timestamp'][df.index[-1]], ZoneInfo(key='Europe/Madrid'))
            print(f"Batch first timestamp: {first_timestamp}")
            print(f"Batch last timestamp: {last_timestamp}")

            if first_timestamp < start_time:
                # print("Skip")
                continue
            else:
                batch_list = df.to_numpy().tolist()
                observation_list += batch_list
                chunks += 1
                if last_timestamp > end_time:
                    # print("That's Enough")
                    break
                print("Keep Logging Data")

        print(f"Number of chunks logged: {chunks}")
        print(f"Number of lines: {len(observation_list)}")
        print(f"Sample line: {observation_list[0]}")

        observation_timestamps = [sublist[0] for sublist in observation_list]
        start_idx = bisect_left(observation_timestamps, start_timestamp)
        end_idx = bisect_left(observation_timestamps, end_timestamp)
        if end_idx == len(observation_list):
            end_idx -= 1
        s_timestamp = datetime.fromtimestamp(observation_list[start_idx][0], ZoneInfo(key='Europe/Madrid'))
        e_timestamp = datetime.fromtimestamp(observation_list[end_idx][0], ZoneInfo(key='Europe/Madrid'))
        print(f"Observation period first timestamp: {s_timestamp}")
        print(f"Observation period last timestamp: {e_timestamp}")

        current_list = observation_list[start_idx:end_idx + 1]
        if i == 0:
            with open(f"ts{test_scenario}_" + zone + freq + '-alldata' + '.pkl', 'wb') as f:
                pickle.dump(current_list, f)
        else:
            with open(f"ts{test_scenario}_" + zone + freq + '-alldata' + '.pkl', 'ab') as f:
                pickle.dump(current_list, f)

        final_chunks += chunks

    print(f"Total chunks: {final_chunks}")
