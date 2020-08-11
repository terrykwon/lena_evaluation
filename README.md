# Code for 'Evaluating the LENA System for Korean'

This repository accompanies our paper, *Evaluating the LENA System for Korean*.
The data used in evaluation is not yet public and this repository is meant to be a reference only for the time being.

The original LENA transcripts are CHAT (`.cha`) files, having been exported from the LENA software.
The human transcripts are TextGrids (`.TextGrid`).

`evaluation.py` contains the methods to parse these transcripts into a common data structure, as well as methods to calculate classification accuracy, and also to extract features such as the word and turn count.
`results.ipynb` contains the high-level code used to generate the results and figures used in the paper.


## Build

The recommended Python version is 3.8.
Dependencies can be installed with `pip` (possibly within a virtual environment).

```
pip install -r requirements.txt
```

In order to calculate the morpheme count for Korean text, the `Mecab` library must additionally be installed locally.
If not, an error will be thrown by the `konlpy` library when a dependent method is called.
Installation instructions can be found in https://konlpy.org/en/latest/install/.