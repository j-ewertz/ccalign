import whisperx
import pandas as pd
from aligner import tokenize_text
from utils import execute_multiprocessing
import json
import os
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import logging
import torchaudio
torchaudio.set_audio_backend("soundfile")
# ignore warnings and logging due to whisperx import
logging.basicConfig(level=logging.ERROR)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)


def apply_whisperx(row: pd.Series,
                   batch_size: int=16,
                   device: str="cuda",
                   dtype: str="float16"):
    
    """Function transcribes a audio file using WhisperX.
    The path to the audio-file is given as an argument"""
    
    # create dict that contains files to be safed
    files_to_safe = {}
    dir = os.path.dirname(row['path_audio'])
    id = row['id']
    path_whisper = os.path.join(dir, f'{id}_whisper.json')
    path_whisperx = os.path.join(dir, f'{id}_whisperx.json')

    # use whisperAI to transcribe audio
    model = whisperx.load_model(
        "base",
        device=device,
        compute_type=dtype,
        language='en'
        )

    whisper_result = model.transcribe(
        row['path_audio'],
        language="en",
        batch_size=batch_size
        )
    
    files_to_safe['whisper'] = {
        'data': whisper_result,
        'path': path_whisper
        }
    
    # clean tokens before apply whisperx
    for i, segment in enumerate(whisper_result['segments']):
        whisper_result['segments'][i]['text'] = ' '.join(
            tokenize_text(segment['text'], tokens_only='true')
            )
    
    # use whisperX to worl-level align the output
    model_a, metadata = whisperx.load_align_model(
        language_code='en',
        device=device
        )
    
    whisperx_result = whisperx.align(
        whisper_result['segments'],
        model_a,
        metadata,
        row['path_audio'],
        device
        )
    
    files_to_safe['whisperx'] = {
        'data': whisperx_result,
        'path': path_whisperx
        }
    
    # safe files
    for file in files_to_safe.values():
        # serializing json
        json_info = json.dumps(file['data'], indent=4)
        
        with open(file['path'], "w") as json_object:
            json_object.write(json_info)
        
    # put processed call into queue
    return {
        'id': id,
        'path_whisper': path_whisper,
        'path_whisperx': path_whisperx
        }



def execute_whisperx(df: pd.DataFrame,
                     batch_size_whisper: int=16,
                     num_processes_whisperx: int=2):
    
    # execute speech-to-text using multiprocessing
    if num_processes_whisperx > 1:
        results = execute_multiprocessing(
                df=df,
                func=apply_whisperx,
                num_processes=2,
                timeout=120,
                groupby=False,
                func_kwargs={
                    'batch_size': batch_size_whisper,
                    'device': "cuda",
                    'dtype': "float16"
                })
    
    # execute speech-to-text process using one process 
    else:
        results = df.apply(apply_whisperx, axis=1).to_list()
    
    # merge infos to original dataframe and return
    df_results = pd.DataFrame(results)
    return df.merge(df_results, on='id')

