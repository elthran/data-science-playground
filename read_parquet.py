from pyspark.sql import SparkSession
import pyspark.sql.functions as f

global_spark = None

# pd.set_option("display.max_rows", 500)
# pd.set_option("display.max_columns", 50)
# pd.set_option("display.width", 1000)


def get_configured_spark():
    global global_spark
    if global_spark is None:
        global_spark = SparkSession.builder \
            .config("spark.sql.session.timeZone", "UTC") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .appName("HeliosETL") \
            .getOrCreate()
    return global_spark


df = get_configured_spark().read \
   .parquet("part-00000-f4a8f9fd-b0ad-4169-ae29-8c9fc2eca4e8-c000.snappy.parquet")

# df = df.withColumn('Time', f.unix_timestamp(df['client_time']))
# df = df.withColumn('ArrivalTime',
#                    f.from_unixtime(f.unix_timestamp(df['server_time'], 'UTC'), 'dd/MMM/yyyy:HH:mm:ss +SSSS'))
#

df.show(700, truncate=False)
for dtype in df.dtypes:
    print(dtype)