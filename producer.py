
from kafka import KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from time import sleep
import json
import os


FILE_PATH = "test.csv"
TOPIC_NAME = "project"
HOST = "localhost:9092"

spark = SparkSession.builder \
  .appName("DataStreamPublisher") \
  .getOrCreate()

df = spark.read.csv(FILE_PATH, inferSchema=True, header=True)
print(df)
print('Importing dataset done')
print('Splitting Done')

df_to_json =  df.toJSON().collect()


producer=KafkaProducer(bootstrap_servers=[HOST],
              api_version=(0,11,5))

print("Sending messages to topic:", TOPIC_NAME)
for elem in df_to_json:
  message = elem.encode('utf-8')
  producer.send(TOPIC_NAME, message)
  sleep(5)
