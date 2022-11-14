
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import gcsfs
import json

from typing import List


def write_metadata(shard_lengths: List[int], data_dir: str):
    features = tfds.features.FeaturesDict({
        'image': tfds.features.Image(),
        'labels': tfds.features.Tensor(shape=(-1,), dtype=tf.int32),
        'label_probs': tfds.features.Tensor(shape=(-1,), dtype=tf.float32),
    })

    split_infos = [
        tfds.core.SplitInfo(
            name='train',
            shard_lengths=shard_lengths,  # Num of examples in shard0, shard1,...
            num_bytes=0,  # Total size of your dataset (if unknown, set to 0)
        ),
    ]

    # https://tensorflow.google.cn/datasets/external_tfrecord
    tfds.folder_dataset.write_metadata(
        data_dir=data_dir,
        features=features,
        split_infos=split_infos,
    )

def main(args):
    shard_lengths = []
    gcs_file_system = gcsfs.GCSFileSystem(project=args.project)
    tfrecord_paths = sorted(tf.io.gfile.glob(os.path.join(args.data_dir, '*.tfrecord')))
    total_tfrecord_paths = len(tfrecord_paths)
    for i, tfrecord_path in enumerate(tfrecord_paths):
        gcs_file_system.rename(tfrecord_path, os.path.join(args.data_dir, f"CoyoLabeled300m-train.tfrecord-{i:05}-of-{total_tfrecord_paths:05}"))
    json_paths = sorted(tf.io.gfile.glob(os.path.join(args.data_dir, '*.json')))
    for path in json_paths:
        with gcs_file_system.open(path) as f:
            json_dict = json.load(f)
        shard_lengths.append(json_dict['successes'])

    write_metadata(shard_lengths, args.data_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()
    
    main(args)