


## Download the metadata
* Download metadata files from Huggingface Dataset
  ```bash
  # install git lfs
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt-get install git-lfs
  git lfs install
  
  # download coyo-700m
  git clone https://huggingface.co/datasets/kakaobrain/coyo-700m
  ```

## Download the images with img2dataset
* [img2dataset](https://github.com/rom1504/img2dataset) can easily download and store to [webdataset](https://github.com/webdataset/webdataset) or tfrecord format for large-scale distributed training.


### Download using Google Cloud Dataproc
* [Dataproc](https://cloud.google.com/dataproc) is a fully managed and highly scalable service for running Apache Spark and 30+ open source tools. 
  With dataproc, you can easily configure multi-node clusters and process large amounts of data quickly.
* Copy `dataproc-initialiation.sh` to your gs bucket. 
  It contains commands to initialize the environment when each node is launched. (e.g, `pip install img2dataset`)
  ```bash
  gsutil cp dataproc-initialization.sh gs://${YOUR_GS_BUCKET}/dataproc/dataproc-initialization.sh
  ```
* Download metadata and upload it to Google Cloud Storage
  ```
  for i in {00000..00127}; do wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-$i-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet -O - | gsutil cp - gs://${YOUR_GS_BUCKET}/dataset/coyo-700m/parquet/part-$i-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet; done
  ```
* Create Dataproc Cluster
    ```bash
    gcloud dataproc clusters create coyo-700m \
        --project=${YOUR_PROJECT_NAME} \
        --region=${YOUR_REGION} \
        --zone=${YOUR_ZONE} \
        --master-machine-type=n2-standard-16 \
        --num-workers=2 \
        --worker-machine-type=n2-standard-16 \
        --num-secondary-workers=8 \
        --secondary-worker-boot-disk-size=100 \
        --image-version=2.0-ubuntu18 \
        --scopes='https://www.googleapis.com/auth/cloud-platform' \
        --properties='yarn:yarn.nodemanager.user-home-dir=/var/lib/hadoop-yarn' \
        --initialization-actions=gs://${YOUR_GS_BUCKET}/dataproc/dataproc-initialization.sh
    ```
    * This command creates 10 nodes for the dataproc cluster
      * 1 master machine
      * 2 primary workers
      * 8 secondary workers (preemptible/spot instances)

* Run/Submit PySpark Job
    ```bash
    gcloud dataproc jobs submit pyspark --cluster=coyo-700m dataproc-img2dataset.py -- \
        --url_list=gcs://${YOUR_GS_BUCKET}/dataset/coyo-700m/parquet \
        --input_format="parquet" \
        --url_col="url" \
        --caption_col="text" \
        --output_format=webdataset \
        --output_folder=gcs://${YOUR_GS_BUCKET}/dataset/coyo-700m/webdataset \
        --distributor="pyspark" \
        --processes_count=1 \
        --thread_count=64 \
        --image_size=384 \
        --retries=1 \
        --min_image_size=200 \
        --max_aspect_ratio=3 \
        --resize_only_if_bigger=True \
        --resize_mode="keep_ratio" \
        --skip_reencode=True \
        --save_additional_columns='["clip_similarity_vitb32","clip_similarity_vitl14","nsfw_score_opennsfw2","nsfw_score_gantman","watermark_score","aesthetic_score_laion_v2"]' \
        --enable_wandb=False
    ```
    * For a detailed description of the arguments, see [img2dataset#API](https://github.com/rom1504/img2dataset#api).

### Download using your own cluster
  * Please see [img2dataset/distributed_img2dataset_tutorial](https://github.com/rom1504/img2dataset/blob/main/examples/distributed_img2dataset_tutorial.md) for more details.

## Missing images
  * COYO-700M was made in 2021, unfortunately, so many images may be gone now.
  * As of August 2022, only about 85% of all images are available for download.
