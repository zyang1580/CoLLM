import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.rec_base_dataset_builder import RecBaseDatasetBuilder
# from minigpt4.datasets.datasets.laion_dataset import LaionDataset
# from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset

from minigpt4.datasets.datasets.rec_datasets import MovielensDataset, MovielensDataset_stage1, AmazonDataset, MoiveOOData, MoiveOOData_sasrec, AmazonOOData, AmazonOOData_sasrec

# @registry.register_builder("movielens")
# class MovielensBuilder(RecBaseDatasetBuilder):
#     train_dataset_cls = MovielensDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/movielens/default.yaml",
#     }

#     def build_datasets(self):
#         # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
#         logging.info("Building datasets...")
#         self.build_processors()

#         build_info = self.config.build_info
#         storage_path = build_info.storage

#         datasets = dict()

#         if not os.path.exists(storage_path):
#             warnings.warn("storage path {} does not exist.".format(storage_path))

#         # create datasets
#         dataset_cls = self.train_dataset_cls
#         datasets['train'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'train')],
#         )
#         try:
#             datasets['valid'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'valid_small2')])
#             datasets['test'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'test')])
#         except:
#             pass

        

#         return datasets

# @registry.register_builder("amazon")
# class AmazonBuilder(RecBaseDatasetBuilder):
#     train_dataset_cls = AmazonDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/amazon/default.yaml",
#     }

#     def build_datasets(self):
#         # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
#         logging.info("Building datasets...")
#         self.build_processors()

#         build_info = self.config.build_info
#         storage_path = build_info.storage

#         datasets = dict()

#         if not os.path.exists(storage_path):
#             warnings.warn("storage path {} does not exist.".format(storage_path))

#         # create datasets
#         dataset_cls = self.train_dataset_cls
#         datasets['train'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'train')],
#         )
#         try:
#             datasets['valid'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'valid_small')])
#             #0915
#             datasets['test'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'test')])
#         except:
#             print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
#             raise FileNotFoundError("file not found.")
#         return datasets


@registry.register_builder("movie_ood")
class MoiveOODBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = MoiveOOData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/movielens/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
            if evaluate_only:
                datasets['test_warm'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test_warm_cold=warm')])

                datasets['test_cold'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test_warm_cold=cold')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets


@registry.register_builder("movie_ood_sasrec")
class MoiveOODBuilder_sasrec(RecBaseDatasetBuilder):
    train_dataset_cls = MoiveOOData_sasrec

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/movielens/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets



@registry.register_builder("amazon_ood")
class AmazonOODBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = AmazonOOData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/amazon/default.yaml",
    }
    def build_datasets(self, evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
            if evaluate_only:
                datasets['test_warm'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test=warm')])

                datasets['test_cold'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test=cold')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets


@registry.register_builder("amazon_ood_sasrec")
class AmazonOODBuilder_sasrec(RecBaseDatasetBuilder):
    train_dataset_cls = AmazonOOData_sasrec

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/amazon/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets
