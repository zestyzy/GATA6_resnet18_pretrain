"""
Teacher model package for generating pseudo-labels using a 3D ResNet‑50 backbone.

This package contains a simple 3D ResNet implementation and associated
utilities for reading NIfTI data, training the network and producing
pseudo‑labels on unlabelled datasets.  The design aims to keep code
modular so that you can reuse individual components without having to
tangle everything together in a single file.  See the individual
modules for details.

Note: In order to actually load NIfTI files you will need to have
``nibabel`` installed in your Python environment.  If you do not have
network access you can obtain nibabel as a wheel and install it
manually, or swap the implementation in ``teacher_model/dataset.py``
for your own NIfTI reader.
"""

from .resnet3d import ResNet3D, generate_resnet18
from .dataset import NiftiDataset
from .train import train_teacher
from .evaluate import evaluate_teacher
from .inference import generate_pseudolabels