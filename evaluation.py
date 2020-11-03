from pyannote.core import Segment, notebook, Annotation

import os
from collections import defaultdict

import numpy as np
import pylangacq as pla
import pandas as pd

import parselmouth
import textgrid

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from konlpy.tag import Hannanum, Mecab


METADATA_PATH = 'data/clip_data.csv'
CHAT_PATH = 'data/chats/'
TEXTGRID_PATH = 'data/textgrids/'


def lena_chat_to_dict(filename_or_chat, child_subcategories=False):
    """ Reads a .cha file that is the output of LENA.

    This method manually parses each utterance and extracts the
    timestamp, speaker, and phonological fragments(?) such as "&=vocalization".

    # Arguments
        filename_or_chat: full path of the .cha file, or the read .cha file itself.

    # Returns
        a dictionary in the format of
                { 'CHI': [((start_ms, end_ms), 'utterance_type'), ... ],
                  'SIL': [...], }
    """
    if type(filename_or_chat) == str:
        cha_reader = pla.chat._SingleReader(filename_or_chat)
        utterances = cha_reader.utterances(clean=False)
    else:
        utterances = filename_or_chat

    tiers = defaultdict(list)

    utterance_map = { # appended to class name if child_subcategories=True
        '&=vocalization': '_voc',
        '&=vfx': '_vfx',
        '&=crying': '_cry'
    }

    for u in utterances:
        speaker = u[0]
        split = u[1].split()
        if len(split) == 2:
            utterance_type = split[0]
            if child_subcategories:
                speaker = '{}{}'.format(speaker, utterance_map[utterance_type])
        else:
            utterance_type = ''
        timestamp = split[-1][:-1].split('_')[-2:]
        timestamp = tuple(int(t) for t in timestamp)

        tiers[speaker].append((timestamp, utterance_type))

    return tiers


def textgrid_to_dict(filename, child_subcategories=False):
    """ Converts textgrid to dict of the same format as `lena_chat_to_dict()`.
    Since the timestamps are in originally in seconds, a conversion to ms by
    rounding down is performed.

    Note: excludes point tier.
    """
    grid = textgrid.TextGrid()
    grid.read(filename)

    subcategories = { # label inconsistencies
        'V':    '_v',
        'VO':   '_vo',
        'F':    '_f',
        ' V':   '_v',
        'F\t':  '_f',
        ' F':   '_f',
        ' ':    '_vo',
        'Ｆ':    '_f',
        'VＯ':   '_vo',
    }

    d = defaultdict(list)

    for tier in [t for t in grid if type(t) != textgrid.PointTier]:
        for interval in tier:
            subcategory = ''
            # an '' interval is empty. else it would be the transcription, 'V', etc.
            if child_subcategories and tier.name == 'Child':
                if interval.mark != '' and not interval.mark.isspace() and interval.mark not in subcategories:
                    subcategory = '_vo' # is vocalization, e.g. 엄마
                elif interval.mark in subcategories:
                    subcategory = subcategories[interval.mark]

            if interval.mark != '' and not interval.mark.isspace(): 
                start_ms = int(interval.bounds()[0] * 1000)
                end_ms = int(interval.bounds()[1] * 1000)

                tier_name = '{}{}'.format(tier.name, subcategory)
                
                d[tier_name].append(((start_ms, end_ms), interval.mark))
    return d


def remap(dictionary, mappings):
    """ Remap dictionary using new mappings.

    Note that the order of items will not be preserved when two keys are combined.
    """
    result = defaultdict(list)

    for tier in dictionary:
        result[mappings[tier]].extend(dictionary[tier])

    return result


def dict_to_frames(tiers, default_class, consider_overlapped=None):
    """ Converts a dictionary of {'tier': [((time_start, time_end), 'mark'), ...]}
    to per-frame format.

    Assumes that LENA does not overlap intervals. However, overlaps are possible in
    human annotations. When any two classes in `consider_overlapped`, the resulting
    frame wil be labelled 'Overlap'.

    If two classes overlap and are not in `consider_overlapped`, then priority will
    be given to human voices over 'Other', 'Silence', or 'TV'. However, priority
    among humans is arbitrary.

    Note: timestamps are in ms, but the granularity of LENA timestamps are 10ms.

    Params:
    - default_class: when there are no intervals that map to that frame (typically would
      be 'silence')
    """
    # print('dict_to_frames', default_class, consider_overlapped)
    frame_length = 10

    frames = [default_class for x in range(30000)] # 5 minutes

    for tier in tiers:
        if tier == default_class:
            continue
        for row in tiers[tier]:
            start_ms, end_ms = row[0]
            
            if len(frames) < int(end_ms / frame_length):
                num_additional_frames = int(end_ms / frame_length) - len(frames)
                frames.extend([default_class for i in range(num_additional_frames)])
                
            for t in range(int(start_ms / frame_length), int(end_ms / frame_length)):
                if frames[t] == default_class: # no overlap
                    frames[t] = tier
                # specified overlap
                elif (consider_overlapped is not None and 
                        tier in consider_overlapped and
                        (frames[t] in consider_overlapped or frames[t] == 'Overlap')):
                    frames[t] = 'Overlap'
                else: # unspecified overlap
                    # print('UnspecifiedOverlap with', frames[t], 'tier', tier)
                    frames[t] = 'UnspecifiedOverlap'
                    # if tier not in ['Other', 'Silence', 'TV', 'Noise']:
                    #     frames[t] = tier
    return frames


def clip_to_frames_single(clip_number, lena_mappings, human_mappings, consider_overlapped, frame_length=10):
    """ Returns the frames for both LENA and human annotations, which can be used to calculate a confusion matrix.
    """
    df = pd.read_csv(METADATA_PATH, index_col='ClipNumber')
    its_filename = df.loc[clip_number].ProcessingFile
    chat_filename = 'e{}.cha'.format(its_filename.split('.')[0])
    textgrid_filename = 'Clip{}.TextGrid'.format(clip_number)
    
    print('Testing accuracy of {} vs {}'.format(chat_filename, textgrid_filename))

    lena_dict = lena_chat_to_dict(os.path.join(CHAT_PATH, chat_filename))
    textgrid_dict = textgrid_to_dict(os.path.join(TEXTGRID_PATH, textgrid_filename))

    lena_dict = remap(lena_dict, lena_mappings)
    textgrid_dict = remap(textgrid_dict, human_mappings)

    # Here we handle overlaps
    # LENA intervals have no overlap.
    lena_frames = dict_to_frames(lena_dict,
            default_class=lena_mappings['SIL'])
    human_frames = dict_to_frames(textgrid_dict,
            default_class=human_mappings['Silence'],
            consider_overlapped=consider_overlapped)
    
    # Extract the relevant 5 minutes from LENA frames
    start_time = df.loc[clip_number].StartTimeS
    end_time = start_time + 300 # 5 minutes
    start_index = int(start_time * 1000 / frame_length)
    end_index = int(end_time * 1000 / frame_length)
    
    lena_frames_sub = lena_frames[start_index : end_index]
    
    # Sanity check
    if (len(lena_frames_sub) != len(human_frames)):
        print(len(lena_frames_sub))
        print(len(human_frames))
        raise Exception('Length of two frame lists are different')
        
    y_lena = [y for y in lena_frames_sub]
    y_human = [y for y in human_frames]
    
    return y_lena, y_human


def clip_to_frames_all(lena_mappings, human_mappings, consider_overlapped, frame_length=10):
    """ Returns the accumulated labelled frames for all 60 clips.
    """
    total_lena = []
    total_human = []

    for i in range(1, 61):
        y_lena, y_human = clip_to_frames_single(i, lena_mappings, human_mappings, 
                consider_overlapped, frame_length)
        total_lena.extend(y_lena)
        total_human.extend(y_human)
    
    return total_lena, total_human


def dict_to_annotation(tier_dict, silence_class):
    """ Converts a dictionary of speech tiers to the `Annotation` data structure used in pyannote libraries.
    """
    annotation = Annotation()
    for tier in tier_dict:
        if tier in silence_class: # Annotations don't require an explicit silence class (this is the default).
            continue
        for (time_start, time_stop), _ in tier_dict[tier]:
            segment = Segment(float(time_start)/1000, float(time_stop)/1000) # ms to seconds
            annotation[segment] = tier
        
    return annotation


def clip_to_annotations(clip_number, lena_mappings, human_mappings):
    """ Returns (human_annotation, lena_annotation)
    """
    df = pd.read_csv(METADATA_PATH, index_col='ClipNumber')

    its_filename = df.loc[clip_number].ProcessingFile
    chat_filename = 'e{}.cha'.format(its_filename.split('.')[0])
    textgrid_filename = 'Clip{}.TextGrid'.format(clip_number)

    lena_dict = lena_chat_to_dict(os.path.join(CHAT_PATH, chat_filename))
    textgrid_dict = textgrid_to_dict(os.path.join(TEXTGRID_PATH, textgrid_filename))
    
    # remap
    lena_dict = remap(lena_dict, lena_mappings)
    textgrid_dict = remap(textgrid_dict, human_mappings)

    # set default (silence) class
    lena_annotation = dict_to_annotation(lena_dict, lena_mappings['SIL'])
    human_annotation = dict_to_annotation(textgrid_dict, human_mappings['Silence'])

    start_time = df.loc[clip_number].StartTimeS
    end_time = start_time + 300 # 5 minutes
    
    # The crop doesn't begin at 0, but at start_time, so we need to shift it left.
    lena_cropped = lena_annotation.crop(Segment(start_time, end_time))
    lena_annotation_shifted = Annotation()
    for segment, track, label in lena_cropped.itertracks(yield_label=True):
        shifted_segment = Segment(segment.start - start_time, segment.end - start_time)
        lena_annotation_shifted[shifted_segment, track] = label

    return human_annotation, lena_annotation_shifted


def count_conversational_turns(filename, included):
    grid = textgrid.TextGrid()
    grid.read(filename)
    
    ct_index = -1
    for (i, tier) in enumerate(grid): # find index of the CT tier... pretty stupid
        if tier.name == 'CT':
            ct_index = i
            
    point_tier = grid[ct_index]
    
    return len([p for p in point_tier if p.mark in included])


def count_child_vocalizations(filename, excluded):
    grid = textgrid.TextGrid()
    grid.read(filename)
    
    ct_index = -1
    for (i, tier) in enumerate(grid):
        if tier.name == 'Child':
            ct_index = i
            
    tier = grid[ct_index]
    
    count = 0
    for interval in tier:
        if interval.mark not in excluded:
            count += 1
    
    return count


def get_total_noise_duration(filename):
    grid = textgrid.TextGrid()
    grid.read(filename)

    noise_tier = grid[grid.getNames().index('Noise')]

    sum = 0
    for interval in noise_tier.intervals:
        if interval.mark != '':
            sum += interval.duration()

    return sum


def get_total_tv_duration(filename):
    grid = textgrid.TextGrid()
    grid.read(filename)

    tv_tier = grid[grid.getNames().index('TV')]

    sum = 0
    for interval in tv_tier.intervals:
        if interval.mark != '':
            sum += interval.duration()

    return sum


def count_word_whitespace(clip_no, path=None, included_tiers=['Female', 'Female2', 'Male', 'Male2']):
    ''' If the path is included, clip_no is ignored (so should be None). 
        If not, then the main transcript is used.
    '''
    if path is None:
        filename = 'Clip{}.TextGrid'.format(clip_no)
        filepath = os.path.join(TEXTGRID_PATH, filename)
    else:
        filepath = path

    count = 0

    transcripts = textgrid_to_dict(filepath)
    for tier in included_tiers:
        for timestamps, utterance in transcripts[tier]:
            split = utterance.split() # split by whitespace
            # print(split)
            count += len(split)

    return count


def count_morphemes_mecab(clip_no, included_tiers=['Female', 'Female2', 'Male', 'Male2']):
    count = 0
    filename = 'Clip{}.TextGrid'.format(clip_no)
    filepath = os.path.join(TEXTGRID_PATH, filename)
    transcripts = textgrid_to_dict(filepath)
    for tier in included_tiers:
        for timestamps, utterance in transcripts[tier]:
            # print(utterance)
            # morphemes = Hannanum().pos(utterance)
            morphemes = Mecab().pos(utterance)
            filtered = [m for m in morphemes if m[1] != 'SF'] # filter out symbols
            # print(filtered)
            count += len(filtered)

    return count


def calculate_frame_ier(y_human, y_lena, speech_tiers, skip_overlap=True):
    """ Given frame-level annotations, calculates the components of the identification
        error rate.
    """
    false_alarms = 0
    misses = 0
    confusions = 0
    total = 0 # total frames of speech in ground truth
    correct = 0

    overlap_count = 0

    for i in range(len(y_human)):
        if skip_overlap and y_human[i] == 'Overlap': # skip overlap
            # print('skip overlap')
            overlap_count += 1
            continue

        if y_human[i] in speech_tiers:
            total += 1

        if y_human[i] == 'Silence' and y_lena[i] in speech_tiers:
            false_alarms += 1
        elif y_human[i] in speech_tiers and y_lena[i] == 'Silence':
            misses += 1
        elif y_human[i] in speech_tiers and y_lena[i] in speech_tiers and y_human[i] != y_lena[i]:
            # print(y_human[i], '!=', y_lena[i])
            confusions += 1

        if y_human[i] in speech_tiers and y_lena[i] in speech_tiers and y_human[i] == y_lena[i]:
            correct += 1

    # print('false alarm', false_alarms)
    # print('total', total)
    # print('correct', correct)
    # print('misses', misses)
    # print('confusions', confusions)
    # print('der',  (false_alarms + misses + confusions) / total )
    # print('overlap count', overlap_count)

    return false_alarms, misses, confusions, total


def total_adult_speech_duration(lena_mappings, human_mappings):
    for i in range(1, 61):
        human_annot, lena_annot = clip_to_annotations(i, lena_mappings, human_mappings)
        print(lena_annot.label_duration('Female') + lena_annot.label_duration('Male'))


def mask_dialogue(filename, output_dir):
    ''' Replaces each Korean character with an "x"
        and saves the file.
    '''
    print(filename)
    grid = textgrid.TextGrid()
    grid.read(filename)

    for tier in grid:
        if type(tier) == textgrid.PointTier:
            continue
        for interval in tier.intervals:
            # if re.search(u'[\u3131-\ucb4c]', interval.mark):
                # print(interval.mark)
            interval.mark = re.sub(u'[\uac00-\ud7af]', 'x', interval.mark)

    grid.write(os.path.join(output_dir, os.path.basename(filename)))