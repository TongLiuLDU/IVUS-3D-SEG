import torch
from nnunetv2.training.nnUNetTrainer.SlidingStripFormer import StripFormer3D_UNETR_PP
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json
from nnunetv2.utilities.crossval_split import generate_crossval_split
import numpy as np
from torch._dynamo import OptimizedModule

class nnUNetTrainer_StripFormer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
        self.enable_deep_supervision = False
        
        
    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    ):
        """构建 StripFormer3D 网络架构。"""

        from dynamic_network_architectures.initialization.weight_init import (  # noqa: WPS433
            InitWeights_He,
        )

        window_sizes = [[3, 3, 3], [19, 3, 3], [21, 3, 3], None]

        model = StripFormer3D_UNETR_PP(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            input_size=[192, 128, 128],
            window_sizes=window_sizes,
            dims=[32, 64, 128, 256],
            depths=[2, 3, 5, 2],
            num_heads=[4, 4, 8, 8],
            mlp_ratios=[4, 4, 4, 4],
            do_ds=False,
            FFN_type="ConvFFN",
        )

        model.apply(InitWeights_He(1e-2))
        return model

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                         identifiers=None,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                
                # 写上自己的划分结果
                splits[self.fold]['train'] = ["140439-Exam1_001","185118-Exam1_002","185118-Exam2_003","20250328103000-Exam1_004","20250328104623-Exam1_005",
                                              "20250328104623-Exam2_006","20250331095622-Exam2_007","20250331095622-Exam3_008","216661-Exam1_009","222290-Exam1_010",
                                              "222290-Exam2_011","234135-Exam1_012","234135-Exam2_013","273343-Exam2_014","296674-Exam1_015",
                                              "296674-Exam2_016","296674-Exam3_017","317277-Exam3_018","327555-Exam1_019","362152-Exam1_020",
                                              "362152-Exam2_021","373028-Exam1_022","373028-Exam3_023","373028-Exam4_024",
                                              "375045-Exam1_029","375045-Exam2_030","375045-Exam3_031","375045-Exam4_032","375045-Exam6_033",
                                              "375045-Exam9_034","375187-Exam1_035","375187-Exam2_036","375462-Exam3_038",
                                              "6212928-Exam2_039","6212942-Exam1_040","6212942-Exam2_041","X375068-Exam1_042","X375068-Exam2_043","X375068-Exam3_044","x222833-Exam1_045","x222833-Exam2_046"]
                
                splits[self.fold]['val'] = ["373065-Exam1_025","373065-Exam2_026","375015-Exam2_027","375015-Exam3_028","375214-Exam1_037"]
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.enable_deep_supervision:
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod

            mod.decoder.deep_supervision = enabled
        else:
            pass