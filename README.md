# Sparse Logit LMM with correlated noise

This code implements a classification model for a Linear Mixed Model (LMM) with assumed correlated noise. 
The classifier is trained with Coordinate Ascent Stochastic Variational Inference .
The structured noise is modelled with a Sparse Gaussian Process, 
the theoretical and mathematical background can be found in my [Thesis] (https://gitlab.tubit.tu-berlin.de/lenz3000/Sparse-Probit-GP-LMM/blob/master/Thesis.pdf)

## Usage
The Algorithm can be called from the command line
'python3 main.py --help' gives out the possible options.

### Using your own dataset
-    If you want to use your own dataset, you will have to put it in an adequate format:
    It has to be a dict loadable with numpy.load(\<PATH\>) and has to contain the entries 'test', 'train' and optionally 'name' and 'side_info'.
    The first two have to be lists of length 2 where the first entry (X) are the data-points and the second (y) are the labels.
    The data-matrix X has to have the dimensionality _dxn_, where _d_ is the dimensionality and _n_ the number of samples and the labels y can be a list or matrix.
    The 'name' entry is the name of the dataset and 
    'side_info' is a dict with entries 'train' 
    and 'test' each containing the side-information matrices with dimensionalities 
    _d'xn_, where _d'_ is the number of side-information features.

-   To train call 'python3 main.py -dp <PATH-TO-DATASET>'. The learner will train on the dataset


### Models
The possible models are shown in this table, the models can be elected by giving a list of the chosen models to the option 'qw-types'.
E.g. "MF Laplace" exists, but not "MAP Gaussian".
|    approximation \ prior on weights           | Laplace           | Horseshoe  |  Gaussian |
| ------------- |:-------------:| -----:|  -----:|
| MF (Mean Field)      | yes | yes |  yes |
| MV (Multivariate)      | yes | yes |  yes |
| MAP | yes (ADMM) | no |  no |
| MAP iterative | yes dimensionwise coordinate Ascent | no |  no |

The 