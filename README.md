# Code for 'Evaluating the LENA System for Korean'

This repository accompanies our upcoming paper, **Evaluating the LENA System for Korean** (McDonald et al., 2020).
The data used in evaluation is not yet public and this repository is meant to be a reference only for the time being.

The original LENA transcripts are CHAT (`.cha`) files, having been exported from the LENA software.
The human transcripts are TextGrids (`.TextGrid`).

`evaluation.py` contains the methods to parse these transcripts into a common data structure, as well as methods to calculate classification accuracy, and also to extract features such as the word and turn count.
The `results.ipynb` Jupyter notebook contains the high-level code used to generate the results and figures used in the paper, such as confusion matrices.

Since the output classes of LENA differ from the human transcripts, diarization / identification evaluation is done by mapping both to the a common set of classes.
The mappings are defined in `mappings/` as JSON files, and enable convenient experimentation with different options.


## Build

The recommended Python version is 3.8.
Dependencies can be installed with `pip` (possibly within a virtual environment).

```
pip install -r requirements.txt
```

In order to calculate the morpheme count for Korean text, the `Mecab` library must additionally be installed locally.
If not, an error will be thrown by the `konlpy` library when a dependent method is called.
Installation instructions can be found in https://konlpy.org/en/latest/install/.