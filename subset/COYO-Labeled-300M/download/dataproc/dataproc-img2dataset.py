# https://github.com/rom1504/img2dataset/blob/main/examples/distributed_img2dataset_tutorial.md

import img2dataset
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName('coyo-labeled-300m')
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .config('spark.driver.maxResultSize', '8192M')  # default: 4096m (dataproc)
        .config('spark.driver.memory', '8192M')         # default: 8192m (dataproc)
        .config('spark.executor.cores', 1)              # default: 8 (dataproc)
        .config('spark.executor.instances', 16)         # default: 2 (dataproc)
        .config('spark.executor.memory', '3G')          # default: 27426m (dataproc)
        .config("spark.task.maxFailures", "100")
        .getOrCreate()
    )

    return spark


if __name__ == '__main__':
    spark = create_spark_session()
    img2dataset.main()
