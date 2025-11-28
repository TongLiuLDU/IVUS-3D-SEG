import os
import shutil
from pathlib import Path
from typing import List

from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, save_json
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import numpy as np




def convert_ultrasound():
    out_dir = os.path.join(nnUNet_raw,"Dataset789_ultrasound")
    
    channel_names={
        0: "Ultrasound",
    }
    
    labels = {
        "background": 0,
        "EEM":1,
        "lumen":2,
    }
    
    generate_dataset_json(
        output_folder=out_dir,
        channel_names=channel_names,
        labels=labels,
        file_ending=".nii.gz",
        num_training_cases=46
    )



if __name__ == "__main__":
    convert_ultrasound()


