import pandas as pd
from aligner import execute_alignment
from transcribe_audio import execute_whisperx
import torch

df_alignment = pd.read_parquet(r'package_testing/df_2_local.parquet')

device, dtype = ("cuda", "float16") if torch.cuda.is_available() else ("cpu", "float32")

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
    num_processes=2,
    calls_per_core=150
    )
