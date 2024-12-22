import time
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import multiprocessing as mp
from typing import Callable, Optional, List, Dict, Any
import regex as re
import inflect    

def execute_multiprocessing(df: DataFrame,
                            func: Callable,
                            num_processes: int,
                            timeout: int = 120,
                            groupby: Optional[bool] = False,
                            func_args: Optional[List[Any]] = None,
                            func_kwargs: Optional[Dict[str, Any]] = None):

    # define structure to process dataframe
    if groupby:
        # change dataframe to list of dataframes
        to_process = df.groupby(['tic', 'callid'], as_index=False)
        to_process = [to_process.get_group(x) for x in to_process.groups]
    else:
        # dont modify original dataframe
        df = df.copy()
        df['mp_processed'] = False
        to_process = df

    # initialize variables
    active_processes = {}
    queue = mp.SimpleQueue()
    results = []
    processed = 0
    total_len = len(to_process)
    pbar = tqdm(total=total_len)


    def queue_wrapper(func, queue, *args, **kwargs):
        result = func(*args, **kwargs)
        queue.put(result)


    def kill_process(process):
        process.terminate()
        process.join(timeout=3)
        process.close()


    def kill_processes(active_processes, wait, pbar):
        # get results from queue to get right status
        while not queue.empty():
            results.append(queue.get())
        
        # check how long processes are alive
        for process in list(active_processes):
            
            process_waiting = wait
            
            # check if processed finished
            if not process.is_alive():
                pbar.update(1)
                del active_processes[process]
                continue
            
            start = active_processes[process]
            alive_dur = time.time() - start
            
            # kill process after timeout
            if alive_dur > timeout:
                kill_process(process)
                del active_processes[process]
            else:
                # join last processes
                while process_waiting:
                    time.sleep(2)
                    while not queue.empty():
                        results.append(queue.get())
                        time.sleep(0.1)
                    exit_code = process.exitcode
                    if exit_code is not None:
                        pbar.update(1)
                        kill_process(process)
                        process_waiting = False
            
            if wait:
                while not queue.empty():
                    results.append(queue.get())
                    time.sleep(0.1)

    while processed < total_len:
        
        time.sleep(0.1)
        
        # kill processes if timeout is passed
        kill_processes(active_processes, False, pbar)
        # check num of running processes
        running_processes = len(active_processes)
        
        # start new processes
        to_start = num_processes - running_processes \
            if num_processes - running_processes < len(to_process) else len(to_process)
        
        for _ in range(to_start):
            
            # define chunk to process
            if isinstance(to_process, list):
                task_arg = to_process[0]
                del to_process[0]
            else:
                # get first row that is not processed
                task_arg = to_process[to_process['mp_processed']==False].iloc[0]
                # set row as processed
                to_process.loc[task_arg.name, 'mp_processed'] = True
            
            # create arguments tuple
            if isinstance(func_args, list):
                args = [task_arg] + func_args
            elif func_args:
                args = [task_arg, func_args]
            else:
                args = [task_arg]
            
            # start process
            process = mp.Process(target=queue_wrapper, args=(func, queue, *args), kwargs=func_kwargs)
            process.start()
            active_processes[process] = time.time()
            
            processed += 1
            
    # join processes and get queue results
    else:
        kill_processes(active_processes, True, pbar)
        if isinstance(results[0], pd.DataFrame):
            results = pd.concat(results)

    # close pbar
    pbar.close()
    print(f'{func} multiprocessing finished')
    return results


def tokenize_text(text: str, tokens_only: bool=True, sep: str=' '):
    """
    Function that tokenizes a text.
    If tokens_only is True, it returns only the cleaned tokens,
    otherwise it return a tuple of the cleaned token and the original token.
    """
    
    # initialize inflect engine
    num_engine = inflect.engine()
    
    token_list = []
    
    # sub % with 'percent'
    text = re.sub(r'\%', ' percent', text)
    
    # sub $ with 'dollar'
    text = re.sub(r'\$', 'dollar ', text)
    
    # sub € with 'euro'
    text = re.sub(r'\€', 'euro ', text)
    
    tokens = text.split(sep)
    # iterate through tokens
    for _, token in enumerate(tokens):
        
        # continue if no charakter is in token
        # or token is only encoded unicode
        if len(token) == 0 or not \
            re.search(r'[\d\w]', token):
            continue
        
        # use placeholders for numbers
        if re.search(r'\d', token):  
            # delete point at the end, e.g. 2016.
            clean_token = re.sub(r'\.$', '', token)
            
            # convert arabic number to string, e.g. 57 to fifty seven
            try:
                clean_token = num_engine.number_to_words(clean_token).lower()
            except Exception as e:
                clean_token = 'numeric_placeholder'
        else:
            clean_token = token.lower()
        
        # sub inword separators with space
        inword_sep = list(re.finditer(r'(?<=\w)[\-\\\/](?=\w)', clean_token))
        if len(inword_sep) > 0:
            for sep in inword_sep:
                clean_token = re.sub(re.escape(sep.group()), ' ', clean_token).lower()
            
        # delete special charakters
        clean_token = re.sub(r'[^\w\s]', '', clean_token).lower()

        # add special charakters to last token
        if len(clean_token) == 0 and len(token_list) > 0:

            token_list[-1] = (token_list[-1][0], token_list[-1][1] + ' ' + token)
            continue

        token_list.append((clean_token, token))
    
    if tokens_only:
        return [token[0] for token in token_list]
    else:
        return token_list

