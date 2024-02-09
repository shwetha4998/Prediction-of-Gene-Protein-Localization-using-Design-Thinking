# Databricks notebook source
#BELIEF NETWORK(NAIVE BAYES)
#Import necessary libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

#Load the dataset

# COMMAND ----------

df = (sqlContext.read.format("csv").
  option("header", "true").
  option("nullValue", "NA").
  option("inferSchema", True).
  load("/FileStore/tables/data151/Genes_relation-3.data"))

# COMMAND ----------

#Display the dataset in the form of dataframe table

# COMMAND ----------

display(df)

# COMMAND ----------

#Replace the '?'value as null

# COMMAND ----------

data =df.na.replace('?', None)

# COMMAND ----------

#Fill the empty values as '0'

# COMMAND ----------

df1 = data.fillna(0)

# COMMAND ----------

#Display the new dataframe table

# COMMAND ----------

display(df1)

# COMMAND ----------

# Define the categorical columns

# COMMAND ----------

categorical_cols = ["Essential", "Class", "Complex", "Phenotype", "Motif", "Chromosome","Localization"]

# COMMAND ----------

#Index the categorical columns

# COMMAND ----------

#indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(data) for col in categorical_cols]
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="skip") for col in categorical_cols]

# COMMAND ----------

#Feature creation

# COMMAND ----------

feature_cols = ["Essential_index", "Class_index", "Complex_index","Phenotype_index","Motif_index" ,"Chromosome_index"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# COMMAND ----------

#Using Naive bayes classifier

# COMMAND ----------

nb = NaiveBayes(labelCol="Localization_index", featuresCol="features", smoothing=1.0, modelType="multinomial")

# COMMAND ----------

#Define the pipeline

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers + [assembler, nb])

# COMMAND ----------

# Split the dataset into training and test sets

# COMMAND ----------

(trainingData, testData) = df1.randomSplit([0.7, 0.3], seed=123)

# COMMAND ----------

#Training the model (Naive Bayes)

# COMMAND ----------

model = pipeline.fit(trainingData)

# COMMAND ----------

#Predictions on Testing data

# COMMAND ----------

predictions = model.transform(testData)

# COMMAND ----------

#Evaluate the model and find the accuracy

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol="Localization_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# COMMAND ----------

#Plot the accuracy using Bar graph

# COMMAND ----------

import matplotlib.pyplot as plt

# Accuracy obtained from the evaluation
# Plotting the bar chart
plt.figure(figsize=(8, 6))
plt.bar(["Accuracy"], [accuracy], color='blue')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.ylim(0, 1)  # Set y-axis limit to ensure the accuracy is properly displayed
plt.show()

# COMMAND ----------

#NEURAL NETWORK
# Import necessary libraries

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# COMMAND ----------

#Load the dataset

# COMMAND ----------

data = (sqlContext.read.format("csv").
  option("header", "true").
  option("nullValue", "NA").
  option("inferSchema", True).
  load("/FileStore/tables/data151/Genes_relation-3.data"))

# COMMAND ----------

#Replace the '?'value as null

# COMMAND ----------

data1 =data.na.replace('?', None)

# COMMAND ----------

#Display the dataset in the form of Dataframe table

# COMMAND ----------

display(data1)

# COMMAND ----------

# Define the categorical columns

# COMMAND ----------

cat_cols = ["Essential", "Class", "Complex", "Phenotype", "Motif", "Chromosome", "Localization"]

# COMMAND ----------

#Index the categorical columns

# COMMAND ----------

indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="skip") for col in cat_cols]
indexers.append(StringIndexer(inputCol="Localization", outputCol="label"))

# COMMAND ----------

#Feature creation

# COMMAND ----------

feature_cols = ["Essential_index", "Class_index", "Complex_index", "Phenotype_index", "Motif_index", "Chromosome_index"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# COMMAND ----------

# Split the dataset into training and test sets

# COMMAND ----------

(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=123)

# COMMAND ----------

# Define the Multilayer Perceptron Classifier

# COMMAND ----------

layers = [len(feature_cols), 10, 5, len(data.select("Localization").distinct().collect())]  # Input layer, 2 hidden layers, output layer
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=123)

# COMMAND ----------

# Define the pipeline

# COMMAND ----------

pipeline = Pipeline(stages=indexers + [assembler, mlp])

# COMMAND ----------

# Train the model

# COMMAND ----------

model = pipeline.fit(trainingData)

# COMMAND ----------


# Make predictions

# COMMAND ----------

predictions = model.transform(testData)

# COMMAND ----------


# Evaluate the model

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy1 = evaluator.evaluate(predictions)
print("Accuracy:", accuracy1)

# COMMAND ----------

#Plot the accuracy using Bar graph

# COMMAND ----------

import matplotlib.pyplot as plt
# Plotting the bar chart
plt.figure(figsize=(8, 6))
plt.bar(["Accuracy"], [accuracy1], color='green')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.ylim(0, 1)  # Set y-axis limit to ensure the accuracy is properly displayed
plt.show()

# COMMAND ----------

#Define the Random Forest Classifier

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=5, maxBins=256, seed=123)

# COMMAND ----------


# Define the pipeline

# COMMAND ----------

pipeline = Pipeline(stages=indexers + [assembler, rf])

# COMMAND ----------

# Train the model

# COMMAND ----------

model = pipeline.fit(trainingData)


# COMMAND ----------

# Make predictions

# COMMAND ----------

predictions = model.transform(testData)

# COMMAND ----------


# Evaluate the model

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy2 = evaluator.evaluate(predictions)
print("Accuracy:", accuracy2)

# COMMAND ----------

import matplotlib.pyplot as plt

# Accuracy obtained from the evaluation
#accuracy = 0.85  # Replace with your actual accuracy value

# Plotting the bar chart
plt.figure(figsize=(8, 6))
plt.bar(["Accuracy"], [accuracy2], color='red')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.ylim(0, 1)  # Set y-axis limit to ensure the accuracy is properly displayed
plt.show()

# COMMAND ----------

#naive_bayes_accuracy = 0.85  # Replace with the actual accuracy of Naive Bayes

# Accuracy of Random Forest classifier
# Plotting the bar graph
classifiers = ['Naive Bayes','Neural Network','Random Forest']
accuracies = [accuracy, accuracy1,accuracy2]

plt.bar(classifiers, accuracies, color=['blue', 'green','red'])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Comparison of Classifier Accuracies')
plt.ylim(0, 1)  # Set y-axis limit to ensure the accuracies are properly displayed
plt.show()

# COMMAND ----------

#How can we make predictions using the attribute ’Localization’?

# COMMAND ----------

predictions.select("Localization", "prediction").show()

# COMMAND ----------

#Visualize the column Localization

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# COMMAND ----------

pandas_df = df1.toPandas()

# COMMAND ----------

class_counts = pandas_df['Localization'].value_counts()

# COMMAND ----------

plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Localization')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
