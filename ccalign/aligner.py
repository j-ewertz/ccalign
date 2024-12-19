import sys
import json
import regex as re
from pydub import AudioSegment
from Levenshtein import distance as levenshtein_distance
from difflib import SequenceMatcher
import csv
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import numpy as np
import time
import inflect
import gc
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


def tokenize_text(text: str, tokens_only: bool=True, sep: str=' '):
    """
    Function that tokenizes a text.
    If tokens_only is True, it returns only the cleaned tokens,
    otherwise it return a tuple of the cleaned token and the original token
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



def compare_strings(s1, s2):
    """
    Function that compares two strings
    and returns the levenshtein distance and similarity.
    """
    
    distance = levenshtein_distance(s1, s2)
    
    similarity = SequenceMatcher(None, s1, s2).ratio()
    
    return distance, similarity


        
@dataclass
class Node:
    transcript_ori: str
    transcript: str
    whisper_word: str
    whisper_index: int
    transcript_index: int
    frontier_index: int
    state: str
    start: float
    end: float
    area_hit: bool
    multiple_word_hit: bool
    distance: float
    similarity: float
    parid: int
    sentid: int
    marker: str
    correct_par: bool = False
    correct_sent: bool = False
    score: Optional[float] = None

    def __repr__(self) -> str:
        return (f"Node(transcript_ori={self.transcript_ori!r}, "
                f"transcript={self.transcript!r}, "
                f"whisper_word={self.whisper_word!r}, "
                f"start={self.start})")


class StackFrontier():
    
    def __init__(self):
        """create frontier"""
        self.frontier = []
    
    
    def __repr__(self):
        return str(self.frontier)
    
    
    def __len__(self):
        return len(self.frontier)
    
    
    def add(self, node):
        """Add note to frontier"""
        self.frontier.append(node)
    
        
    def remove(self, num_to_remove):
        """
        Method that removes x last nodes
        """
        self.frontier = self.frontier[:-num_to_remove]
     
        
    def last_node(self):
        """
        Return last element if it exists
        """ 

        # return last node if it exists
        if self.frontier:
            return self.frontier[-1]
        
        # return starting node
        else:
            return -1
    
    
    def first_node(self):
        """
        Returns first element
        """
        
        return self.frontier[0]
    
    
    def return_segment(self, parid):
        """
        Returns all node of a specific paragraph
        """
        return [node for node in self.frontier
                if node.parid == parid]
    
    
    def check_track(self, steps=2, frontier_index=None):
        """Check if last nodes are aligned correctly."""
        
        if frontier_index != None:
            nodes_to_check = self.frontier[frontier_index-steps:frontier_index]
            negative_index = frontier_index-steps <= 0
        # if no position tuple is given, use last x nodes
        else:
            nodes_to_check = self.frontier[-steps:]
            negative_index = len(self.frontier) < steps
        
        correct_matches = []
        
        for node in nodes_to_check:
            if node.state == True:
                correct_matches.append(1)
        
        # return true if all matches are correct or it is the beginning
        if len(correct_matches) == len(nodes_to_check) or negative_index:
            return True
        elif len(correct_matches) == 0:
            return False
        else:
            return None
    
    
    def postprocess(self):
        """
        Function processes the stack and changes nodes declared as false
        """
        
        # use two iterations to clean the stack
        for _ in range(2):
            for i, node in enumerate(self.frontier):
                
                # set nodes to True, if the distance and similarity are below thresholds
                if node.distance <= 3 and node.similarity > .5:
                    node.state = True
                
                # set nodes to True, if the word before and after are correct
                # and the distance is not above threshold
                if i > 0 and i < len(self) - 1 and self.frontier[i-1].state == True and self.frontier[i+1].state == True:
                    
                    # set to true if word before and after are not area matches
                    if self.frontier[i-1].area_hit == False and self.frontier[i+1].area_hit == False:
                        node.state = True
                        
                    # set to true if distance is below threshold
                    elif node.similarity >= .5:
                        node.state = True
                    
        
    def multiple_area_hits(self, number):
        """Returns whisper positions and frontier position of multiple area hits if existing"""
      
        nodes = self.frontier[-number:]
        
        for node in nodes:
            if isinstance(node.area_hit, list):
                if len(node.area_hit) > 1:
                    return node
                
        return None           
    

    def debug_info(self,
                   start,
                   end,
                   to_csv=False,
                   csv_path=None,
                   export_audio=False,
                   path_to_audio=None):
        
        """Function that takes a start and end point
        and shows debugging information of the corresponting stack extraction.
        If 'to_csv' is True, a csv file is exported,
        else information is printed on terminal.
        """
        
        header = ['state', 'transcript', 'whisper',
                  'parid', 'par_status',
                  'sentid', 'sent_status',
                  'area_hit', 'm_w_h', 'dist', 'sim',
                  'fid', 'whisper_i', 'transcript_ori',
                  'start', 'end', 'score']
        
        if to_csv:
            formatter = ['<0' for i in range(len(header))]
            
            # initialize a csv writer
            csvfile = open(csv_path, 'w', encoding='UTF8', newline='')
            csv_writer = csv.writer(csvfile)
                
        else:
            formatter = ['<8', '<15', '<15',
                         '^6', '^6', '^6',
                         '^6', '^10', '^8',
                         '^8', '^6', '^6',
                         '^6', '<15', '^6'
                         '^6', '^6']

        if end > len(self):
            end = len(self)
        
        # clean audio_examples folder
        if export_audio and path_to_audio:
            existing_files = os.listdir(r'audio_examples/word_level')
            for file in existing_files:
                os.remove(os.path.join(r'audio_examples/word_level', file))
            # load audio file    
            audio = AudioSegment.from_mp3(path_to_audio)
        
        # enhance by 1 because of header
        for i in tqdm(range(end - start)):    
            
            # use header in the first iteration
            if i == 0:
                info = header
            else:
                node = self.frontier[start]
                info = [str(node.state), node.transcript, node.whisper_word,
                        node.parid, node.correct_par,
                        node.sentid, node.correct_sent,
                        str(node.area_hit), str(node.multiple_word_hit), node.distance,
                        round(node.similarity, 2), node.frontier_index,
                        node.whisper_index, node.transcript_ori,
                        node.start, node.end, node.score]
                
                # increase start by 1
                start += 1
            
            row = [f"{info[_i]:{formatter[_i]}}" for _i in range(len(header))]
        
            if to_csv:
                csv_writer.writerow(row)
            
            else:
                row = ''.join(row)
                print(row)

            # export cutted words if export_audio is True and node.state is True
            if export_audio and path_to_audio and node.state and i > 0: 
                transcript_word = node.transcript
                start_timing, end_timing = node.start, node.end
                cutted_audio = audio[start_timing*1000:end_timing*1000]
                # check if word already exists in dir
                count = [file[:-4] for file in os.listdir(r'audio_examples/word_level')].count(transcript_word)
                # if word already exists rename file
                if count > 0:
                    cutted_audio.export(f'audio_examples/word_level/{transcript_word}_{count+1}.mp3')
                else:
                    cutted_audio.export(f'audio_examples/word_level/{transcript_word}.mp3')


    def check_multiple_positions(self, position, nums_to_check):
        """Function that take a position and checks the corrects alignment 
        in this and the following 2 positions"""
        
        add_position = list(range(nums_to_check))
        positions_list = [position + x for x in add_position]
        position_checks = [self.check_track(2, position) for position in positions_list]
        
        return True if True in position_checks else False
    
    
    def stats(self, markers, print_stats):
        """
        Function that checks how many nodes, paragraphs and sentences are
        aligned correctly. Adds information to the stack and returns
        statistics
        """
        
        # initialize needed variables
        last_node = self.last_node()
        stats_par_sents = {}
        total_stats = {'correct_paragraphs': None,
                       'correct_sentences': None,
                       'correct_nodes': None}
        
        # iterate through paragraphs
        last_par_num = last_node.parid
        for parid in range(last_par_num+1):
            
            par_stats = {'par':None, 'sent':{}}
            
            # extract nodes of the segment
            nodes = self.return_segment(parid)
            # skip segment if no nodes are found
            if len(nodes) == 0:
                continue
            # check beginning of the segment
            par_stats['par'] = self.check_multiple_positions(nodes[0].frontier_index, 3)
            
            # iterate through sentences
            for sentence_number in range(nodes[-1].sentid + 1):

                # check the beginning of each sentence
                sent_position = [node.frontier_index for node in nodes if node.sentid == sentence_number][0]
                par_stats['sent'][sentence_number] = self.check_multiple_positions(sent_position, 3)
                    
            stats_par_sents[parid] = par_stats
        
        # add info to stack frontier
        for node in self.frontier:
            par_num = node.parid
            sent_num = node.sentid

            par_stats = stats_par_sents[par_num]
            if par_num + 1 < len(stats_par_sents):
                # check if par_num + 1 exists
                if par_num + 1 in stats_par_sents.keys():
                    next_par_stats = stats_par_sents[par_num+1]
                # if segment is empty, use next one
                else:
                    next_par_stats = stats_par_sents[par_num+2]
            else:
                next_par_stats = False
                par_end = True
                sent_end = True
                
            # extract start and end info of par and sent
            sent_start = par_stats['sent'][sent_num]
            if sent_num + 1 < len(par_stats['sent']):
                sent_end = par_stats['sent'][sent_num+1]
            elif next_par_stats:
                sent_end = next_par_stats['sent'][0]

            par_start = par_stats['par']
            if next_par_stats:
                par_end = next_par_stats['par']
            
            if stats_par_sents == stats_par_sents:
                node.correct_sent = True if sent_start and sent_end else False
                node.correct_par = True if par_start and par_end else False
                    
        # create overall stats
        # extract relevant nodes
        relevant_nodes = [node for node in self.frontier if node.marker in markers]
        
        # calc rations
        uncorrect_nodes = len([node for node in relevant_nodes if node.state == False])
        correct_nodes = len(relevant_nodes) - uncorrect_nodes
        
        relevant_par = set([node.parid for node in relevant_nodes])
        correct_par, correct_sents, num_sents = 0, 0, 0
        
        for par in relevant_par:
            par_nodes = [node for node in relevant_nodes if node.parid == par]
            if par_nodes[0].correct_par:
                correct_par += 1
                
            for sent in range(par_nodes[-1].sentid + 1):
                num_sents += 1
                sent_node = [node for node in par_nodes if node.sentid == sent][0]
                if sent_node.correct_sent:
                    correct_sents += 1
        
        # add information to statistics-dict
        total_stats['correct_paragraphs'] = correct_par/len(relevant_par)
        total_stats['correct_sentences'] = correct_sents/num_sents
        total_stats['correct_nodes'] = correct_nodes/(correct_nodes+uncorrect_nodes)
        
        if print_stats:
            print(f'correct paragraphs: {correct_par/len(relevant_par)}')
            print(f'correct sentences: {correct_sents/num_sents}')
            print(f"{'correct aligned words:':<28}{correct_nodes:^10}")
            print(f"{'uncorrect aligned words:':<28}{uncorrect_nodes:^10}")
            print(f"{correct_nodes/(correct_nodes+uncorrect_nodes):.2f}")
        
        return total_stats
    
    
class WhisperOutput():

    def __init__(self, whisperx_path, row):

        # read whisperx file
        with open(whisperx_path) as whisperx_f:
            self.whisperx = json.load(whisperx_f)

        self.segments = self.whisperx['segments']
        
        # define token and position list
        self.tokens = []
        self.position_list = []
        self.max_lengths = len(self.segments)
        self.alignment_status = None
        self.id = row['id']

        # iterate through whisper segments and create tokens
        for _, segment in enumerate(self.segments):

            tokens = tokenize_text(segment['text'])
            self.tokens.extend(tokens)
            
            if len(self.position_list) == 0:
                self.position_list.append(len(tokens))
            else:
                self.position_list.append(len(tokens)+self.position_list[-1])
        
            # clean segment text
            segment['text'] = ' '.join(tokens)
        
        # perform word_level_alignment
        self.word_level_alignment()                
            
    
    def __len__(self):
        return len(self.tokens)
        
        
    def segment_information(self, position):
        """Function that can take a position or a list of positions and returns segment information"""
        
        position_start = position[0] if isinstance(position, list) else position

        segment_number = self.position_list.index(min(threshold for threshold
                                                    in self.position_list
                                                    if threshold > position_start))

        if segment_number == 0:
            segment_position_start = position_start
        else:
            segment_position_start = position_start - self.position_list[segment_number-1]

        if isinstance(position, list):
            return {segment_number: {segment_position_start: len(position)}}
        else:
            return segment_number, segment_position_start
    
    
    def segment_token(self, segment_i):
        """Method that takes the segment-index and 
        return to segment token and lengths"""

        if segment_i == 0:
            seg_tokens = self.tokens[:self.position_list[segment_i]]
        else:
            seg_tokens = self.tokens[self.position_list[segment_i - 1]
                                        :self.position_list[segment_i]]
        
        return seg_tokens, len(seg_tokens)
        
    
    def clean_timing(self, position_dict, delete_false_entries=True):
        """
        Method that cleans timing information in case of a multiple word hit.
        Takes position dict as an parameter in form of {segment: {starting index:len word}
        """

        # extract key from dict
        i = list(position_dict.keys())[0]

        # iterate through combined words in specific segment i
        for start_i, end_i in position_dict[i].items():
            # decrease end index by one to enable list slicing
            end_i = start_i + end_i -1
            text = ' '.join([word['word'] for word in self.segments[i]['words'][start_i:end_i+1]])
            # extract start timing
            start_timing = self.segments[i]['words'][start_i]['start']
            
            # extract score
            start_score = self.segments[i]['words'][start_i]['score']
            
            # only change last entry of combined words
            if not delete_false_entries:
                # if end is in next segment, move to next segment
                # can only happend if the combined word is found in the transcript
                if len(self.segments[i]['words']) <= end_i:
                    end_i = end_i - len(self.segments[i]['words']) - 1
                    end_timing = self.segments[i+1]['words'][end_i]['end']
                    end_score = self.segments[i+1]['words'][end_i]['score']
                    self.segments[i+1]['words'][end_i] = {'word': text,
                                                          'start': start_timing,
                                                          'end': end_timing,
                                                          'score': (start_score + end_score) / 2}
                else:
                    end_timing = self.segments[i]['words'][end_i]['end']
                    end_score = self.segments[i]['words'][end_i]['score']
                    self.segments[i]['words'][end_i] = {'word': text,
                                                        'start': start_timing,
                                                        'end': end_timing,
                                                        'score': (start_score + end_score) / 2}
                    
            # replace original whisper output with new values
            else:
                end_timing = self.segments[i]['words'][end_i]['end']
                end_score = self.segments[i]['words'][end_i]['score']
                del self.segments[i]['words'][start_i:end_i+1]
                self.segments[i]['words'].insert(start_i, {'word': text,
                                                           'start': start_timing,
                                                           'end': end_timing,
                                                           'score': (start_score + end_score) / 2})
            

    def word_level_alignment(self):
        """
        Method that performs the word level alignment 
        of whisperx and tokenization
        """
        
        # clean segments that have been sorted to original
        for segment_i, segment in enumerate(self.segments):
            
            _, seg_lengths = self.segment_token(segment_i)
            
            # check for segments that whisperx resorted to original
            if len(segment['words']) == 0:
                # add words
                for missing_word_i in range(seg_lengths):
                    segment['words'].insert(missing_word_i, 
                                            {'word': 'resorting_placeholder',
                                             'score': 0})  
        
        # add timing information to all words
        for segment_i, segment in enumerate(self.segments):
            
            for word_i, word in enumerate(segment['words']):

                # clean timing if missing, e.g. for numbers
                keys = word.keys()
                
                # search for start and end of previous or following words
                if not 'start' in keys:
                    start_found, end_found = False, False
                    word_i_down, word_i_up = word_i, word_i
                    segment_i_down, segment_i_up = segment_i, segment_i
                    start, end = None, None
                    # check for start, i_up and i_down as fallbacks
                    i_up, i_down = 0, 0
                    while not start_found and i_up < 100:
                        i_up += 1
                        word_i_down -= 1
                        if abs(word_i_down) < len(self.segments[segment_i_down]['words']):
                            word_before = self.segments[segment_i_down]['words'][word_i_down]
                        else:
                            segment_i_down -= 1
                            word_i_down = -1
                            if segment_i_down >= 0:
                                word_before = self.segments[segment_i_down]['words'][word_i_down]
                            # use the very start if no word could be found
                            else:
                                start, start_found = 0, True
                        if 'end' in word_before.keys():
                            start, start_found = word_before['end'], True
                    # check for end
                    while not end_found and i_down < 100:
                        i_down += 1
                        word_i_up += 1
                        # check if next segment exists
                        if segment_i_up < self.max_lengths and \
                            word_i_up < len(self.segments[segment_i_up]['words']):
                            # use segment if word index exists
                            word_after = self.segments[segment_i_up]['words'][word_i_up]
                        else:
                            segment_i_up += 1
                            word_i_up = 0
                            # check if next segment exists
                            if segment_i_up < self.max_lengths:
                                next_seg_words = self.segments[segment_i_up]['words']
                                word_after = next_seg_words[word_i_up]
                            # if no further segment exists use the total end
                            else:
                                word_after = {'start': self.segments[segment_i_up-1]['end']}
                        if 'start' in word_after.keys():
                            end, end_found = word_after['start'], True
                            
                    # add start and end information to word dict
                    word['start'] = start
                    word['end'] = end
        
        # delete special words
        for segment_i, segment in enumerate(self.segments):
        
            seg_tokens, seg_lengths = self.segment_token(segment_i)
            
            for word_i, word in enumerate(segment['words']):
                
                # delete words without number or char in word
                # or percent or dollar
                if not re.search(r'[\d\w]', word['word']):
                    del segment['words'][word_i]
                
            # check if alignment worked correctly
            if len(segment['words']) != seg_lengths:
                print(f'Error in: {self.tic}, {self.id}')
                print('Single words: ', [word['word'] for word in segment['words']])
                print('Needed tokens: ', seg_tokens)
                self.alignment_status = False
                    
        # return True if everything worked fine
        self.alignment_status = True

   
    def return_word(self, position):
        
        segment_number, segment_position = self.segment_information(position)
        segment = self.segments[segment_number]
        word = segment['words'][segment_position]
        
        # returns word timing in format ('word': ..., 'start': ..., 'end': ...)
        return word
        
    
    def search_word_area(self, position, word, area):
        
        upper = position + area
        lower = 0 if position - area < 0 else position - area
        
        indices = [i for i, whisper_word in enumerate(self.tokens[lower:upper]) if whisper_word == word]
        indices = [i + lower for i in indices]
        
        if len(indices) > 0:
            # add original position
            # original position might be better than area hit
            indices.insert(0, position)
            return indices
        else:
            return None
        
            
class Aligner():
    """A Class that takes a row containing path information and performs the alignment."""
    
    def __init__(self, row):
        
        self.id = row['id']
        
        # define path variables
        speech_sequence_path = row['path_transcript']
        whisperx_path = row['path_whisperx']
        self.audio_path = row['path_audio']
        
        
        self.local_paths = {
            'speech_sq': speech_sequence_path,
            'whisperx': whisperx_path,
            'audio': self.audio_path
            }
                        
        # read transcript
        with open(self.local_paths['speech_sq']) as transcript_f:
            self.speech_sequence = json.load(transcript_f)
        
        # generate whisper object
        self.whisper = WhisperOutput(self.local_paths['whisperx'], row)
        
        # create a stack that keeps track of each aligned word
        self.stack = StackFrontier()
    
    
    def delete_files(self, file_list):
        """
        Function that deletes local files.
        """
        
        for file in file_list:
            os.remove(file)
        
    
    def compare_hits(self, area_hit, paragraph_tokens, transcript_index, include_hit=False, comp_words=3):
        
        # adjust index if word itself sould be in- or excluded
        i = 0 if include_hit else 1
        
        # define dict to store similarity results
        sim_dict = {key: None for key in area_hit}
        
        # iterate through hits
        for whisper_index in area_hit:
            
            # use next three words to define similarity of strings
            # clean if tokens are given in form of tuples
            if isinstance(paragraph_tokens[0], tuple):
                relevant_tokens = [token[0] for token in    # ('thank', 'Thank')
                                    paragraph_tokens[transcript_index+i:transcript_index+i+comp_words]]
            else:
                relevant_tokens = paragraph_tokens
            paragraph_string = ' '.join(relevant_tokens)
            whisper_string = ' '.join(self.whisper.tokens[whisper_index+i:whisper_index+i+comp_words])
            
            _, sim = compare_strings(paragraph_string, whisper_string)
            sim_dict[whisper_index] = sim
        
        # return whisper index (key) with highest similarity
        return max(sim_dict, key=sim_dict.get)
        
        
    def fallback(self, speech_i, transcript_index=0, token_len=5):
        """
        Fallback method that searchs phrases in the whisper text
        and returns the whisper index and transcript index if possible,
        else None, None.
        """

        # use the first index, if word index < 0
        if transcript_index < 0:
            transcript_index = 0
        
        whisper_text = "__".join(self.whisper.tokens)
        transcript_text = self.speech_sequence['speech_parts'][speech_i]['lines'][0]
        transcript_tokens = tokenize_text(transcript_text)
        max_tokens = len(transcript_tokens)
        
        # generate phrases to search for
        for i in range(max_tokens - token_len - transcript_index):
            
            transcript_index += i
            
            if max_tokens >= transcript_index+token_len:
                tokens = transcript_tokens[transcript_index:transcript_index+token_len]
            else:
                # return None if not enough tokens exists in the paragraph
                return None, None
            
            # do not use spaces to verify right match,
            # because multiple-words-hit contain spaces as well
            phrase = "__".join(tokens)
            # add '__' to start and end to prevent finding match
            # within words and returning wrong index
            phrase = "__" + phrase + "__"
            # use C style formatting to add fuzzy components 
            phrase = "(%s){e<=0}" % phrase  #TODO: add fuzzy matching in future version
            matches = list(re.finditer(phrase, whisper_text))
            
            # use match if there is only one match
            if len(matches) == 1:
                
                match = matches[0]
                whisper_index = len(tokenize_text(whisper_text[:match.start()], sep='__'))
                if self.whisper.tokens[whisper_index] == transcript_tokens[transcript_index]:
                    return whisper_index, transcript_index
                         
            # extract best match if multiple matches are found
            elif len(matches) > 1:

                matching_scores = {match:{'similarity': None,
                                          'whisper_index': None} for match in matches}
                for match in matches:
                    
                    start = match.start()
                    whisper_index = len(tokenize_text(whisper_text[:start], sep="__"))
                    # compare the next words of the same paragraph if possible
                    if i + 2*token_len <= max_tokens and whisper_index + 2*token_len <= len(self.whisper):
                        new_transcript_tokens = transcript_tokens[transcript_index+token_len:
                                                                  transcript_index+2*token_len]
                        
                    # use next paragraph if not enough tokens exists in this paragraph
                    else:
                        new_transcript_tokens = transcript_tokens[transcript_index+token_len:]
                        # add tokens from next paragraph(s) 
                        while len(new_transcript_tokens) < token_len:
                            speech_i += 1
                            if speech_i < len(self.speech_sequence['speech_parts']):
                                
                                needed_tokens = token_len - len(new_transcript_tokens)
                                transcript_text_i1 = self.speech_sequence['speech_parts'][speech_i]['lines'][0]
                                transcript_tokens_i1 = tokenize_text(transcript_text_i1)[:needed_tokens]
                                new_transcript_tokens += transcript_tokens_i1
                            else:
                                # break while loop, if not enough tokens exist
                                break     
                    
                    # compare strings if needed conditions are met
                    if len(new_transcript_tokens) == token_len \
                        and whisper_index + 2*token_len <= len(self.whisper):
                        
                        next_phrase_transcript = "__".join(new_transcript_tokens)
                        tokens_whisper = self.whisper.tokens[whisper_index+token_len:whisper_index+2*token_len]
                        next_phrase_whisper = "__".join(tokens_whisper)
                        _, similarity = compare_strings(next_phrase_transcript, next_phrase_whisper)
                        matching_scores[match]['similarity'] = similarity
                    
                    else:
                        # set similarity to zero if no calculation is possible
                        matching_scores[match]['similarity'] = 0 
                        
                # return best match
                match = max(matching_scores, key=lambda x: matching_scores[x]['similarity'])
                whisper_location = len(tokenize_text(whisper_text[:match.start()], sep='__'))
                if self.whisper.tokens[whisper_location] == transcript_tokens[transcript_index]:
                    return whisper_location, transcript_index
                    
        # return None if no match is found
        return None, None
                
        
    def alignment(self, markers, print_stats=False):
        """Algorithm that alignes whisperx output and transcript

        Args:
            dict in the form of {'speech_parts': {'lines': list[str['speaker level text here']],
                                                  'marker': any[str['-PR-', '-Q_A-', '-Q-', '-OP-']],
                                                  'speaker': str[speaker name here],
                                                  'speaker_info': str['speaker information here']}} 

        Returns:
            stack, statistics: returns aligned stack and statistics 
            based on the markers
        """
        
        
        for speech_number, speech_part in enumerate(self.speech_sequence['speech_parts']):
            
            marker = speech_part['marker']
            paragraph_text = speech_part['lines'][0]
            paragraph_tokens = []
            sentence_info = []
            
            # iterate through each sentence of the speech_part
            for sentence in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph_text):
                
                sentence = ' ' + sentence           
                tokens = tokenize_text(sentence, tokens_only=False)
                if len(tokens) == 0:
                    continue

                paragraph_tokens.extend(tokens)
                sentence_info.append(len(paragraph_tokens))
                          
            # if the speech_sequence changes and track is lost, use fallback
            whisper_fallback_index = None
            if speech_number > 0 and self.stack.check_track() is False:
                whisper_fallback_index, transcript_fallback_index = self.fallback(speech_number)
            
            # define starting index
            transcript_index = 0

            # delete empty sentences from sentence_info
            # e.g. [14, 19, 42, 65, 78, 78, 99]
            sentence_info = list(set(sentence_info))
            sentence_info.sort()
            
            # iterate through each word
            # use a while loop to be able to move backwards
            while transcript_index < len(paragraph_tokens):
                
                # define starting variables 
                add_node, distance, similarity, area_hit, multiple_word_hit, alignment_status = True, 0, 0, False, False, False
                
                # define the sencence number
                sentence_number = sentence_info.index(min([threshold for threshold
                                                          in sentence_info if threshold > transcript_index]))
                
                last_node = self.stack.last_node()
                if isinstance(last_node, Node):
                    old_index = last_node.whisper_index
                # start with -1
                else:
                    old_index = last_node
                
                
                # use fallback position if track was lost and fallback was found
                if whisper_fallback_index and transcript_index == transcript_fallback_index:
                    whisper_index = whisper_fallback_index
                    whisper_fallback_index = None
                
                # use the next word if list index is in range
                elif old_index < len(self.whisper.tokens) - 1:
                    whisper_index = old_index + 1
                    
                # use old index if list index is out of range
                else:
                    whisper_index = old_index
                
                # define words
                transcript_word_original = paragraph_tokens[transcript_index][1]
                transcript_word = paragraph_tokens[transcript_index][0]
                whisper_word = self.whisper.tokens[whisper_index]
                
                # check if the next word is correct
                if transcript_word == whisper_word:
                    alignment_status = True
                
                # if the word is not found, check in the area around it
                elif self.whisper.search_word_area(whisper_index, transcript_word, 10):
                    area_hit = self.whisper.search_word_area(whisper_index, transcript_word, 10)
                    # return best hit 
                    whisper_index = self.compare_hits(area_hit, paragraph_tokens, transcript_index)
                        
                    # define whisper word
                    whisper_word = self.whisper.tokens[whisper_index]
                    if whisper_word == transcript_word:
                        alignment_status = True
                    
                # check for combined words,
                # space is used as a placeholder for combined words
                elif ' ' in transcript_word:
                    
                    separated_words = transcript_word.split(' ')
                    
                    # check for beginning if not correct
                    if not separated_words[0] == whisper_word:
                        area_hit = self.whisper.search_word_area(whisper_index, separated_words[0], 4)
                        # return best hit
                        if area_hit:
                            whisper_index = self.compare_hits(area_hit,
                                                              separated_words,
                                                              0,
                                                              include_hit=True,
                                                              comp_words=len(separated_words))
                            # define whisper word
                            whisper_word = self.whisper.tokens[whisper_index]
 
                    multiple_word_hit = [whisper_index]  # list of whisper indices to know start and end
                    for _ in range(1, len(separated_words)):

                        if whisper_index + 1 < len(self.whisper.tokens):
                            whisper_index += 1
                            whisper_word += ' ' + self.whisper.tokens[whisper_index]
                            multiple_word_hit.append(whisper_index)

                    distance, similarity = compare_strings(transcript_word, whisper_word)
                    if similarity >= 0.7:
                        alignment_status == True
                    
                        # clean timing information
                        position_dict = self.whisper.segment_information(multiple_word_hit)
                        self.whisper.clean_timing(position_dict, delete_false_entries=False)
                    
                # if no match exists, use fallback
                elif self.stack.check_track(steps=4) is False and not whisper_fallback_index:
                    distance, similarity = compare_strings(transcript_word, self.whisper.tokens[whisper_index])                    
                    whisper_fallback_index, transcript_fallback_index = self.fallback(speech_number, transcript_index-4)                        
                    # check if fallback index is the actual index
                    # repeat iteration in that case
                    if transcript_index == transcript_fallback_index:
                        # only repeat iteration if both words are equal
                        transcript_index -= 1
                        
                    # check if the fallback hit was before the actual transcript position
                    elif transcript_fallback_index is not None and transcript_fallback_index < transcript_index:
                        # remove last nodes
                        self.stack.remove(transcript_index-transcript_fallback_index)
                        transcript_index = transcript_fallback_index - 1 # -1 because index is raised at the end of the iteration                    
                        add_node = False
                else:
                    distance, similarity = compare_strings(transcript_word, whisper_word)
                        
                # extract timing information
                word = self.whisper.return_word(whisper_index)
                start = word['start']
                end = word['end']
                if 'score' in word.keys():
                    score = word['score']
                else:
                    score = 0

                # add node to stack
                if add_node:
                    self.stack.add(Node(
                        transcript_word_original,
                        transcript_word,
                        whisper_word,
                        whisper_index,
                        transcript_index,
                        len(self.stack),
                        alignment_status,
                        start,
                        end,
                        area_hit,
                        multiple_word_hit, 
                        distance,
                        similarity, 
                        speech_number,
                        sentence_number,
                        marker,
                        score=score
                        ))

                transcript_index += 1
        
        # postprocess the stack
        self.stack.postprocess()
        
        # add par and sentence statistics
        stats = self.stack.stats(markers, print_stats=print_stats)
        
        # add identification info to stats and stack
        stats['id'] = self.id
        
        stack_dict = {'id': self.id, 'stack': self.stack}
        
        # return stack and statistics
        return stack_dict, stats


def postprocess_results(results):
    
    """Function that processes the results after alignment

    Args:
        results (tuple of nested lists): results of align_dataframe()
    """

    start_timing = time.time()
    
    # sep aligned stack and statistics
    df_stacks = [result[0] for result in results]
    stats = [result[1] for result in results]
    # check which rows have been processed
    processed_rows = [result[2] for result in results]
    
    # flatten list of results
    stats = [sublist for result in stats for sublist in result]
    processed_rows = [sublist for result in processed_rows for sublist in result]
    
    # transform stats into dataframe
    df_stats = pd.DataFrame(stats)
    
    # create df to store word level information
    df_word_level = pd.concat(df_stacks)
    
    # free memory
    del df_stacks
    del stats
    gc.collect()
    
    sys.stdout.write('stack_df_created\n')
    
    # create sentence level dataframe
    groupby_cols = ['id', 'parid', 'sentid']
    df_sent = df_word_level.groupby(groupby_cols)['start'].min().reset_index()
    
    merge_dict = {'end': df_word_level.groupby(groupby_cols)['end'].max().reset_index()}
    merge_dict['marker'] = df_word_level.groupby(groupby_cols)['marker'].max().reset_index()
    merge_dict['state'] = df_word_level[groupby_cols + ['correct_sent']].drop_duplicates(groupby_cols)
    merge_dict['start_score'] = df_word_level.sort_values('transcript_index').groupby(groupby_cols, as_index=False).first()[groupby_cols + ['score']]\
        .rename({'score': 'start_score'}, axis=1)
    merge_dict['end_score'] = df_word_level.sort_values('transcript_index').groupby(groupby_cols, as_index=False).last()[groupby_cols + ['score']]\
        .rename({'score': 'end_score'}, axis=1)
    merge_dict['avg_score'] = df_word_level.groupby(groupby_cols, as_index=False)['score'].mean().rename({'score':'avg_score'}, axis=1)
    merge_dict['min_score'] = df_word_level.groupby(groupby_cols, as_index=False)['score'].min().rename({'score':'min_score'}, axis=1)
    merge_dict['max_score'] = df_word_level.groupby(groupby_cols, as_index=False)['score'].max().rename({'score':'max_score'}, axis=1)
    merge_dict['text'] = df_word_level.groupby(groupby_cols)['transcript_ori'].apply(lambda x: ' '.join(x)).reset_index()
    # merge dataframe and delete to free memory
    for key in list(merge_dict):
        df_sent = df_sent.merge(merge_dict[key], on=groupby_cols)
        del merge_dict[key]
        gc.collect()

    end_timing = time.time()
    dur = round((end_timing - start_timing) / 60, 3)
    sys.stdout.write(str(dur))
    
    return df_word_level, df_sent, df_stats, processed_rows


def align_dataframe(df):
    """
    Function that executes the actual alignment using a dataframe as
    a parameter containing needed path variables.
    """
    
    stack_list, stats_list, processed_calls  = [], [], []
    
    for _, row in df.iterrows():
        
        processed_calls.append([row['id']])
        try:
            aligner = Aligner(row=row)
            stack, stats = aligner.alignment(markers=['-Q-', '-Q_A-', '-PR-'], print_stats=False)
            stack_list.append(stack)
            stats_list.append(stats)
        except Exception as e:
            error_str = f"'error in: {row['id']} f{e}\n"
            sys.stdout.write(error_str)

    # create df to store word level information
    df_word_level = pd.DataFrame()
    
    # convert stack list to dict to reduce memory usage
    stack_dict = {i:item for i, item in enumerate(stack_list)}
    del stack_list
    gc.collect()
    
    # add stack information to dataframe
    for key in list(stack_dict):
        stack = stack_dict[key]
        stack_frontier = stack['stack'].frontier
        stack_data = list(node.__dict__ for node in stack_frontier)
        df_stack = pd.DataFrame(stack_data)
        df_stack['id'] = stack['id']
        
        df_word_level = pd.concat([df_word_level, df_stack])
        
        del stack_dict[key]
        gc.collect()
    
    return (df_word_level, stats_list, processed_calls)


def execute_alignment(df, num_processes=False, calls_per_core=100):
    
    if num_processes is False:
        num_processes = mp.cpu_count() - 2
    
    # initialize variables
    if 'aligned' not in df.columns:
        df['aligned'] = False
    
    rows_to_process = df[df['aligned']==False]
    df_stats_total = pd.DataFrame()
    start = time.time()
    iteration = 0
    
    while len(rows_to_process) > 0:
        
        if isinstance(iteration, int):
            iteration += 1
        
        # create a pool of workers
        pool = mp.Pool(processes=num_processes)
        
        # define calls to process
        num_to_process = calls_per_core * num_processes
        iteration_chunks = rows_to_process[:num_to_process]
        
        # split the dataframe
        df_splitted = np.array_split(iteration_chunks, num_processes)
        
        # set performed_iterations as a fixed parameter to allow map()

        # apply pooling using map
        results = pool.map(align_dataframe, df_splitted)
        
        # close the pool and wait with further execution
        # until all subprocesses are completed.
        pool.close()
        pool.join()
        pool.terminate()
        sys.stdout.write('\n\npool terminated successfully\n\n')
        
        # process for results
        df_word_level, df_sent_level, df_stats, processed_rows = postprocess_results(results)
        sys.stdout.write('\n\nresults processed successfully\n\n')

        # add results to dataframes
        df_stats_total = pd.concat([df_stats_total, df_stats])

        # update calls to process
        for processed_row in processed_rows:
            id = processed_row[0]
            df.loc[df['id']==id, 'aligned'] = True
            
        rows_to_process = df[df['aligned']==False]
        
        end = time.time()
        dur = round((end - start) / 60, 3)
        left = len(rows_to_process)
        dur_str = f'\n\nIt took {dur} min to process {num_to_process} calls.\n{left} left to process.\n\n'
        sys.stdout.write(dur_str)
        
        # safe dataframes
        # df.to_pickle(r'df_alignment.pkl')
        
        df_stats_total.to_pickle(r"df_stats.pkl")
        df_word_level.to_pickle(f'df_word_level_{iteration}.pkl')
        df_sent_level.to_pickle(f'df_sent_level_{iteration}.pkl')

        # update starting time
        start = end
        # delete results
        del results, df_word_level, df_sent_level
        gc.collect()
