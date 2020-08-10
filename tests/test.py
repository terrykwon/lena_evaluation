""" Note: uses clip no. 50 as test file, which was chosen because
    it contains many speakers. Version before reannotation.
"""

import unittest
import pylangacq as pla
import parselmouth
import textgrid
import math

from evaluation import lena_chat_to_dict
from evaluation import textgrid_to_dict
from evaluation import dict_to_frames
from evaluation import count_child_vocalizations
from evaluation import count_conversational_turns


class TestConversionMethods(unittest.TestCase):
    chat_filepath = 'tests/data/sample_lena_chat.cha'
    textgrid_filepath = 'tests/data/sample_textgrid.TextGrid'


    def test_lena_chat_to_dict(self):
        x = lena_chat_to_dict(self.chat_filepath)
        all_lena_tiers = {'SIL', 'NOF', 'TVF', 'NON', 'OLN', 'OLF', 'CXF', 
                         'FAN', 'CXN', 'FAF', 'CHN', 'MAF', 'CHF', 'MAN', 'TVN'}
        self.assertTrue(set(x.keys()).issubset(all_lena_tiers))

        # Randomly sampled points
        self.assertTrue(((0, 1180), '') in x['FAF'])
        self.assertTrue(((5164650, 5165530), '') in x['SIL'])
        self.assertTrue(((20449700, 20450060), '&=vocalization') in x['CHN'])
        self.assertTrue(((49724150, 49725160), '') in x['TVN'])
        self.assertTrue(((53175900, 53177380), '') in x['NON'])
        

    def test_textgrid_to_dict(self):
        x = textgrid_to_dict(self.textgrid_filepath)
        some_human_tiers = {'Child', 'Female', 'Male'}
        self.assertTrue(((math.floor(1.716625 * 1000), math.floor(3.0439375 * 1000)), 
                "오빠가 자꾸 때려싸.")  in x['Female']) 
        self.assertTrue(some_human_tiers <= set(x.keys())) # check if subset

        x2 = textgrid_to_dict(self.textgrid_filepath, child_subcategories=True)
        some_human_tiers = {'Child_v', 'Female', 'Male'}
        self.assertTrue(some_human_tiers <= set(x2.keys())) # check if subset
        self.assertTrue('Child' not in x2.keys()) 


    def test_dict_to_frames_lena(self):
        d = lena_chat_to_dict(self.chat_filepath)
        f1 = dict_to_frames(d, default_class='SIL')

        self.assertEqual(f1[115], 'FAF')
        self.assertEqual(f1[516465], 'SIL')


    def test_dict_to_frames_textgrid(self):
        d = textgrid_to_dict(self.textgrid_filepath)
        overlapped = {'Male', 'Male2', 'Female', 'Female2', 'Child'}
        f = dict_to_frames(d, default_class='Silence', 
                consider_overlapped=overlapped)

        self.assertEqual(f[2000], 'Male')
        self.assertEqual(f[2156], 'Overlap')
        self.assertEqual(f[2678], 'Noise')
        self.assertEqual(f[29920], 'Female')


    def test_textgrid_to_annotation(self):
        pass
        # self.assertEqual(1, 1)

    
    def test_cvc_count(self):
        exclude_from_vocalizations = ('F', 'V')
        pred = count_child_vocalizations(self.textgrid_filepath,
                excluded=exclude_from_vocalizations)
        
        true = 6 # manually counted in Praat

        self.assertEqual(true, pred)


    def test_ctc_count(self):
        include_as_turn = ('CA', 'AC', 'CO', 'OC', 'CC2', 'C2C')
        pred = count_conversational_turns(self.textgrid_filepath, included=include_as_turn)
        true = 3 # manually count in Praat
        self.assertEqual(true, pred)



# if __name__ == '__main__':
#     unittest.main()

