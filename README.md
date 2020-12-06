# Code for 'Evaluating the LENA System for Korean'

This repository accompanies our upcoming paper, **Evaluating the LENA System for Korean** (McDonald et al., 2020).

## Data
Our data, coding manual, and paper draft is shared on https://osf.io/uztxr/.

The original LENA transcripts are CHAT (`.cha`) files, having been exported from the LENA software.
The human transcripts are TextGrids (`.TextGrid`).
The spreadsheet `clip_data.xlsx` includes the relevant variables for each of the 60 clips, such as AWC and CVC.

## Analysis code
`evaluation.py` contains the methods to parse the CHAT and TextGrid transcripts into a common data structure, as well as methods to calculate classification accuracy, and also to extract features such as the word and turn count.

Since the output classes of LENA differ from the human transcripts, diarization / identification evaluation is done by mapping both to the a common set of classes.
The mappings are defined in `mappings/` as JSON files, and enable convenient experimentation with different options.

The `results.ipynb` Jupyter notebook contains some of the high-level code used to generate the results and figures used in the paper, such as confusion matrices. 
The remaining errors, error rates, correlations, and graphs of comparisons of LENA and human codings are calculated in `results.R`.


## Build

The recommended Python version is 3.8.
The recommended R version is 3.5.3.
Python dependencies can be installed with `pip` (possibly within a virtual environment).

```
pip install -r requirements.txt
```

In order to calculate the morpheme count for Korean text, the `Mecab` library must additionally be installed locally.
If not, an error will be thrown by the `konlpy` library when a dependent method is called.
Installation instructions can be found in https://konlpy.org/en/latest/install/.
