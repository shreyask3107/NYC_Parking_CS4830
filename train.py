#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from datetime import datetime
from pyspark.sql.types import DateType, IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, IndexToString

spark = SparkSession.builder.getOrCreate()

FILE_PATH = 'gs://bdl2022/trainingdatanyc.csv'

df = spark.read.format('csv').option("header", "true").option("inferSchema", "true").load(FILE_PATH)
# df.columns


# In[ ]:


def preprocessing(df):
    # Dropping columns with many null values
    to_drop = ['From Hours In Effect', 'Intersecting Street', 'Time First Observed', 'Violation Legal Code', 
               'Unregistered Vehicle?', 'Meter Number', 'No Standing or Stopping Violation', 'Hydrant Violation', 
               'Double Parking Violation', 'To Hours In Effect']
    df = df.drop(*to_drop)

    df = df.dropDuplicates()
    
    func =  F.udf(lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())
    df = df.withColumn('Issue_Date2', func(F.col('Issue Date')))
    df = df.drop('Issue Date')
    df = df.withColumnRenamed('Issue_Date2','Issue Date')

    # Deriving new columns from Issue Date
    df = df.withColumn("Issue Day", F.dayofweek(F.col("Issue Date"))).withColumn("Issue Month", F.month(F.col("Issue Date"))).withColumn("Issue Year", F.year(F.col("Issue Date")))
    
    # Filtering rows with Issue year from 2013 to 2017 
    df = df.filter((F.col("Issue Year") > 2012) & (F.col("Issue Year") < 2018))

    # Dividing Violation Time into bins
    def bins(x):
        default_ = 3
        if not x:
            return default_
        hr = x[:2]
        period = x[-1].upper()

        if hr in ['00','01','02','03','12'] and period == 'A':
            return 1
        elif hr in ['04','05','06','07'] and period == 'A':
            return 2
        elif hr in ['08','09','10','11'] and period == 'A':
            return 3
        elif hr in ['12','00','01','02','03'] and period == 'P':
            return 4
        elif hr in ['04','05','06','07'] and period == 'P':
            return 5
        elif hr in ['08','09','10','11'] and period == 'P':
            return 6
        else:
            return default_

    bin_udf = F.udf(bins, IntegerType())
    df = df.withColumn("Time bin", bin_udf(F.col("Violation Time")))

#     Dropping columns which seem to be irrelevant to affect Violation County
    to_drop = ['Summons Number', 'Plate ID', 'Vehicle Expiration Date', 'House Number', 
               'Street Name', 'Vehicle Color', 'Date First Observed', 'Days Parking In Effect', 'Violation Post Code', 
               'Vehicle Year', 'Feet From Curb', 'Issue Date', 'Violation Time', 
               'Issuer Code', 'Vehicle Make']
    df = df.drop(*to_drop)

    default_values = {
        'Registration State': 'NY',
        'Plate Type': 'PAS',
        'Issue Month': 6,
        'Issue Year': 2015,
        'Vehicle Body Type': 'SUBN',
        'Issuing Agency': 'T',
        'Street Code1': 0,
        'Street Code2': 0,
        'Street Code3': 0,
        'Issuer Command': 'T103',
        'Issuer Squad': 'A',
        'Violation In Front Of Or Opposite': 'F',
        'Violation County' : '00000',
        'Violation Description' : "38-Failure to Display Muni Rec",
        'Violation Location' : 54,
        'Law Section' : 459,
        'Sub Division' : 'h1',
        'Violation Code': 35,
        'Issuer Precinct': 56
        }

    df = df.na.fill(default_values)
    return df


# In[ ]:


df = preprocessing(df)
print('Preprocessing Done')


# In[ ]:


print('Train test split')
training_data, test_data = df.randomSplit([0.9, 0.1], seed=42)


# In[ ]:


str_cols = training_data.columns
indexed_features = [col + "_idx" for col in str_cols]
features = list((set(training_data.columns)) - set(str_cols)) + indexed_features
features = list(set(features) - set(['Violation County_idx']))

print('Indexing of categorical features')

featureIndexers = StringIndexer(inputCols = str_cols, outputCols = indexed_features, handleInvalid="keep").fit(training_data)
featureIndexers_save = Pipeline(stages = [featureIndexers]).fit(training_data) 
featureIndexers_save.save("gs://be18b014-bdl/project_final/StringIndex")
training_data = featureIndexers_save.transform(training_data)
assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="keep")
training_data = assembler.transform(training_data)
print('Assembling Done')


# In[ ]:


print('Training a Random Forest model') 
lr = RandomForestClassifier(numTrees=500, maxBins = 10000, labelCol="Violation County_idx", featuresCol="features")
model_save = Pipeline(stages = [lr]).fit(training_data)
model_save.save("gs://be18b014-bdl/project_final/ClassifierRF")
print('Model Training Done')


# In[ ]:


from pyspark.ml import PipelineModel

str_cols = test_data.columns
indexed_features = [col + "_idx" for col in str_cols]
features = list((set(test_data.columns)) - set(str_cols)) + indexed_features
features = list(set(features) - set(['Violation County_idx']))

loadIndexer = PipelineModel.load('gs://be18b014-bdl/project_final/StringIndex')
test_data = loadIndexer.transform(test_data)


# In[ ]:


test_data = model_save.transform(test_data)

evaluatorF1 = MulticlassClassificationEvaluator(
        labelCol="Violation County_idx", predictionCol="prediction", metricName="f1")

evaluatorAcc = MulticlassClassificationEvaluator(
    labelCol="Violation County_idx", predictionCol="prediction", metricName="accuracy")

print('Evaluating')
Accuracy = evaluatorAcc.evaluate(test_data)
F1 = evaluatorF1.evaluate(test_data)
print('Evaluation Done')


# In[ ]:


print(f"F1: {F1:.5f}")
print(f"Accuracy: {Accuracy:.5f}")


# In[ ]:


labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=featureIndexers.labelsArray[6])
test_data = labelConverter.transform(test_data)


# In[ ]:


test_data.select(['predictedLabel', 'Violation County']).show(10)

