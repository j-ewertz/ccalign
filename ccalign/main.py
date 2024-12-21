import pandas as pd
from alignment import execute_alignment
import torch
from audio_transcription import execute_whisperx
import multiprocessing as mp

df_alignment = pd.read_parquet(r'package_testing/df_2_local.parquet')
df_alignment.columns

device, dtype = ("cuda", "float16") if torch.cuda.is_available() else ("cpu", "float32")
cpu_cores = mp.cpu_count()

execute_whisperx(
    df_alignment,
    model="base.en",
    batch_size_whisper=16,
    num_processes_whisperx=2,
    device=device,
    dtype=dtype
    )
        

execute_alignment(
    df_alignment,
    num_processes_alignment=cpu_cores - 2,
    calls_per_core=150
    )


df_sent = pd.read_pickle(r'df_sent_level_1.pkl')
df_stats = pd.read_pickle(r'df_stats.pkl')
