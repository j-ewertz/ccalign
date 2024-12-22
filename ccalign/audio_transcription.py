import whisperx
import pandas as pd
from .utils import tokenize_text, execute_multiprocessing
import json
import os
from datasets import Dataset
from typing import Literal, Union
import logging
import warnings
# TODO: adjust when whisperx pull request #936 is accepted
# new package versions: faster-whisper==1.1.0 and pyannote.audio==3.3.2 
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)


def apply_whisperx(row: pd.Series,
                   model: str="base.en",
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
        model,
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



def execute_whisperx(path_data: Union[pd.DataFrame, Dataset],
                     model: str="large-v2",
                     batch_size_whisper: int=16,
                     num_processes_whisperx: int=2,
                     device: Literal["cuda", "cpu"]="cpu",
                     dtype:str="float16") -> pd.DataFrame:
    
    # convert dataset to dataframe
    if isinstance(path_data, Dataset):
        path_data = pd.DataFrame(path_data)
    
    # execute speech-to-text using multiprocessing
    if num_processes_whisperx > 1:
        results = execute_multiprocessing(
                df=path_data,
                func=apply_whisperx,
                num_processes=2,
                timeout=120,
                groupby=False,
                func_kwargs={
                    'model': model,
                    'batch_size': batch_size_whisper,
                    'device': device,
                    'dtype': dtype
                })
    
    # execute speech-to-text process using single core 
    else:
        results = path_data.apply(apply_whisperx, axis=1).to_list()
    
    # merge infos to original dataframe
    df_results = pd.DataFrame(results)
    return path_data.merge(df_results, on='id')

