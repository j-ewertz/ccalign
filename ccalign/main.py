import pandas as pd
from aligner import execute_alignment
from transcribe_audio import execute_whisperx
import multiprocessing as mp


df_alignment = pd.read_parquet(r'package_testing/df_2_local.parquet')

    
if __name__ == '__main__':
    num_processes = mp.cpu_count() - 2


    execute_alignment(df_alignment, num_processes=num_processes, location='local', calls_per_core=150, debug=False)
