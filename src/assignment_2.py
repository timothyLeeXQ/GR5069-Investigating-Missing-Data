# Databricks notebook source
# MAGIC %md
# MAGIC # Assignment 2: Analysing missing data in F1 Dataset
# MAGIC 
# MAGIC **Objectives:**
# MAGIC 
# MAGIC * Uncover patterns of missing data in datasets
# MAGIC * See how missing data can affect the distribution of variables
# MAGIC * Use and evaluate imputation techniques

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Setup

# COMMAND ----------

# %matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Relevant Packages

# COMMAND ----------

from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from numpy.random import rand
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Mount S3 Bucket to obtain data
# MAGIC 
# MAGIC * Provide AWS Access Keys for authentication
# MAGIC * **DELETE ACCESS KEYS IMMEDIATELY AFTER MOUNTING**
# MAGIC * Provide AWS Bucket Name and mount name
# MAGIC * Expect successful mount - line 8 should produce paths to be used to get data

# COMMAND ----------

# Uncomment, insert credentials, and run to mount
#ACCESS_KEY = ''
# Encode the Secret Key as that can contain '/'
#SECRET_KEY = ''.replace("/", "%2F")
#AWS_BUCKET_NAME_RAW = 'ne-gr5069'
#MOUNT_NAME_RAW = 'ne-gr5069'

#dbutils.fs.mount('s3a://%s:%s@%s' % (ACCESS_KEY, SECRET_KEY, AWS_BUCKET_NAME_RAW),
#                 '/mnt/%s' % MOUNT_NAME_RAW)
#display(dbutils.fs.ls('/mnt/%s' % MOUNT_NAME_RAW))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mount S3 Bucket to Write Data

# COMMAND ----------

AWS_BUCKET_NAME_PROC = 'xql2001-gr5069'
MOUNT_NAME_PROC = 'xql2001-gr5069'

dbutils.fs.mount('s3a://%s:%s@%s' % (ACCESS_KEY, SECRET_KEY, AWS_BUCKET_NAME_PROC),
                 '/mnt/%s' % MOUNT_NAME_PROC)
display(dbutils.fs.ls('/mnt/%s' % MOUNT_NAME_PROC))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define functions to be used throughout

# COMMAND ----------

def fill_dummy_values(df, scaling_factor):
  # Since Matplotlib does not plot missing values, we would have to create 
  # a function that fills dummy variables into the null values of the
  # dataframe
    #create copy of dataframe
    df_dummy = df.copy(deep = True)
    # Iterate over each column
    for col in df_dummy:
        if df_dummy.dtypes[col] != np.object:
          #get column, column missing values and range
          col = df_dummy[col]
          col_null = col.isnull()
          num_nulls = col_null.sum()
          col_range = col.max() - col.min()
        
          #Shift and scale dummy values
          dummy_values = (rand(num_nulls) - 2)
          dummy_values = dummy_values * scaling_factor * col_range + col.min()
        
          #Return dummy values
          col[col_null] = dummy_values
    return df_dummy

def cols_to_int_type(df, col_list):
  # :::::::::::: DESCRIPTION
  # This function is used to change a set of columns in a dataframe to an
  # integer type by looping through through columnsprovided as a list
  #
  # Functionalising this because many of the dfs in the F1 data should be
  # integers, but every variable is imported as string
  #
  # ::::::::: INPUTS
  # 1. df - the dataframe with columns to be changed to int. Should be a 
  #    pyspark.sql dataframe object
  # 2. col_list - a list of strings - each the name of a column in the df
  #    that is to be changed to an integer datatype
  #
  # ::::::::: OUTPUT
  # The dataframe entered as an argument, but with the desired columns
  # cast to the datatype Integer
  #
  for colname in col_list:
    df = df.withColumn(colname, df[colname].cast(IntegerType()))
  return(df)


def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data
  
def RMSE(predict, target):
  #Calculate RMSE
  return np.sqrt(((predict - target) ** 2).mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load in missing data dataset

# COMMAND ----------

df_missing = spark.read.csv('/mnt/ne-gr5069/raw/df_missing.csv', header = True)
display(df_missing)

# COMMAND ----------

display(df_missing.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Change datatype of appropriate columns to int

# COMMAND ----------

df_missing_int_colname = ['_c0',
                          'driverId',
                          'number',
                          'age',
                          'raceId',
                          'lap',
                          'position',
                          'milliseconds']

df_missing = cols_to_int_type(df_missing, df_missing_int_colname)
display(df_missing)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Identify and Isolate Missing values
# MAGIC 
# MAGIC *There are missing values in other rows as well, like driver code (in the last
# MAGIC assignment), but here our focus is on the 3 identified for the assignment -
# MAGIC `milliseconds`, `nationality`, and `age`*
# MAGIC 
# MAGIC I downloaded the raw data and the datatype transformed data from above and had
# MAGIC a look in Excel (shameless, I know, but it's just very comfortable to use 
# MAGIC Excel when one needs to just scroll through a table and have a look).
# MAGIC 
# MAGIC Missing values in the raw data were coded as "NA". This is still true for the
# MAGIC 'nationality' column, which is still string type. For the age and milliseconds
# MAGIC columns, transforming the data to integer type caused non number values to
# MAGIC become blank (e.g. from NA to a blank).
# MAGIC 
# MAGIC The following cells are meant to further explore the data after transformation.
# MAGIC The main goal is to see if we've managed to identify the pattern of all the
# MAGIC missing or invalid values (all missing values are coded NA, not NULL or
# MAGIC NaN, and no more stray missing or invalid vals in age and milliseconds)

# COMMAND ----------

display(df_missing.summary())

# COMMAND ----------

df_missing_unique_nationality = df_missing.groupby('nationality')\
  .count()

display(df_missing_unique_nationality)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Age and millisecond data:**
# MAGIC It seems that the change to integer type has led to some missing data being
# MAGIC identified. summary() now produces 411348 entries for age and 425253 entries
# MAGIC for milliseconds, both less than the total number of rows, 472504.
# MAGIC 
# MAGIC The average, min, and max values of age and milliseconds appear to be
# MAGIC reasonable for F1 drivers and laps. I think it is reasonable to conclude any
# MAGIC data left in the column are valid ages and laptimes.
# MAGIC 
# MAGIC **Nationality data:**
# MAGIC 
# MAGIC It appears the only values that NA is the only tag for missing values.
# MAGIC Everything else seems to be a valid nationality.
# MAGIC 
# MAGIC Although, the overall DF summary suggests that there was no missing data in the
# MAGIC nationality set, that's likely because "NA" was considered a valid string.
# MAGIC 
# MAGIC The groupby operation above shows that there were 47112 NA values, which should
# MAGIC be all missing values in the column. This corresponds to 425392 valid entries.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Three variables, `milliseconds`, `nationality` and `age` have different missingness patterns. Use exploratory data analysis to determine the missingness pattern on each variable, and explain how you reached that conclusion.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Age
# MAGIC Will first try to filter for just the missing data and look for patterns

# COMMAND ----------

df_age_na = df_missing.where(F.col('age').isNull())
display(df_age_na)

# COMMAND ----------

# Check if this is all the missing values
display(df_age_na.summary())

# COMMAND ----------

# Looks like we're good
61156 + 411348

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC By clicking the sort buttons in the filtered data table and looking at the 
# MAGIC summary, I think the missingness pattern is related to the drivers, for 
# MAGIC the following reasons:
# MAGIC 
# MAGIC 1. Other variables appear to have the full range of variation. Only raceId and
# MAGIC driver related variables like driverId appear to have truncated variation. The
# MAGIC filtered dataset only contains race numbers 50-879 as opposed to 1-1030, and 
# MAGIC drivers 30-86 as opposed to 1-848.
# MAGIC 
# MAGIC 2. Sorting the table with only missing age values up and down in databricks
# MAGIC does not do anything for driver related variables. Only Michael Schumacher 
# MAGIC is reflected.
# MAGIC 
# MAGIC I think the restriction in the range of raceIds might be related to the
# MAGIC drivers. As drivers 30-86 probably don't have careers that span the whole 
# MAGIC dataset, they aren't going to have all the raceIds reflected in their data.
# MAGIC Furthermore, looking at the data in Excel (again!) I realised that all Michael
# MAGIC Schumacher's age data is missing, not just those for specific races. So I'm 
# MAGIC going to look into driverIds first and see if that resolves the issue, then
# MAGIC look at races if it does not.

# COMMAND ----------

df_age_na_driverIds = df_age_na.groupby('driverId')\
  .count()
display(df_age_na_driverIds)

# COMMAND ----------

list_age_na_drivers = df_age_na_driverIds.toPandas()['driverId'].tolist()

# COMMAND ----------

df_age_na_replicated = df_missing\
  .where(F.col('driverId').isin(list_age_na_drivers))

display(df_age_na_replicated.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Success! By getting the driverIds within df_age_na dataframe, then
# MAGIC filtering the overall missing dataset to include entries with only these
# MAGIC driverIds, we successfully replicated the df_age_na dataframe in the
# MAGIC dataframe df_age_na_replicated, as can be seen by:
# MAGIC 
# MAGIC 1. Counts are the same for both these dataframes
# MAGIC 2. Age is missing for the entire replicated dataframe.
# MAGIC 3. Descriptive stats for variables like lap, position, and milliseconds
# MAGIC  are identical.
# MAGIC  
# MAGIC  We can thus conclude that the missingness pattern of age in df_missing
# MAGIC  is that age is missing for the drivers with driverIds contained within
# MAGIC  `list_age_na_drivers`.
# MAGIC  
# MAGIC  **Note:** With hindsight having done questions 2 and 3, it is also
# MAGIC  clear that the drivers with missing age data were all over 50 years
# MAGIC  old. Given that the df_missing dataset simply didn't have this data,
# MAGIC  it was impossible to know this without looking at the original data
# MAGIC  first. This leaves 2 possibilities:
# MAGIC  1. All ages above 50 were removed, creating a situation where all
# MAGIC  the ages of drivers in `list_age_na_drivers` were removed, without
# MAGIC  affecting any other driverrs (since they were below 50 years old).
# MAGIC  2. The ages of drivers in `list_age_na_drivers` were intentionally
# MAGIC  removed, and they were all above 50 years old.
# MAGIC  
# MAGIC  1 is the more likely scenario, but without looking at the original
# MAGIC  data until tackling question 2, I was led to believe that 2 was
# MAGIC  what was happening.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Milliseconds
# MAGIC 
# MAGIC Will try the same strategy as was done for age. After all it worked once.

# COMMAND ----------

df_milliseconds_na = df_missing.where(F.col('milliseconds').isNull())
display(df_milliseconds_na)

# COMMAND ----------

display(df_milliseconds_na.summary())

# COMMAND ----------

# Looks good here too
47251 + 425253 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Missingness here appears to be related to position. The full df_missing
# MAGIC dataset has positions from 1-24, whereas this dataset, containing only data 
# MAGIC where milliseconds is missing only has positions from 16 to 24.
# MAGIC 
# MAGIC Other variables appear similar or the same as the full df_missing dataset,
# MAGIC so will look at position first.

# COMMAND ----------

df_milliseconds_na_pos_16_filter = df_missing\
  .filter(F.col('position') >= 16)
display(df_milliseconds_na_pos_16_filter)

# COMMAND ----------

display(df_milliseconds_na_pos_16_filter.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Unfortunately that didn't work out. The dataset df_milliseconds_na_pos_filter
# MAGIC filtered for positions equal to or more than 16, but this produced a
# MAGIC dataframe almost twice the size of our milliseconds missing dataset, with
# MAGIC some milliseconds values not missing.
# MAGIC 
# MAGIC To see the difference between the missing and non missing milliseconds values,
# MAGIC I've filtered the dataset further to provide us with a dataframe with positions
# MAGIC equal to or more than 16, and only non-missing milliseconds values

# COMMAND ----------

df_milliseconds_filled_pos_filter = df_milliseconds_na_pos_16_filter\
  .where(F.col('milliseconds').isNotNull())

display(df_milliseconds_filled_pos_filter)

# COMMAND ----------

display(df_milliseconds_filled_pos_filter.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Comparing our milliseconds missing data to the non-missing milliseconds data
# MAGIC to the data with both, though all with positions >= 16, we see a further
# MAGIC difference in the positions data. 
# MAGIC 
# MAGIC 1. The means of the positions data is smaller in the dataset where all
# MAGIC milliseconds information is missing than in the combined dataset, which is
# MAGIC smaller than in the dataset containing only non-missing milliseconds data
# MAGIC where positions >= 16.
# MAGIC 
# MAGIC 2. Range for non-missing milliseconds data positions is 16-19, not 16-24.
# MAGIC Meaning all positions >=20 are NA. Testing this theory below.

# COMMAND ----------

df_milliseconds_na_pos_20_filter = df_milliseconds_na_pos_16_filter\
  .filter(F.col('position') >= 20)
display(df_milliseconds_na_pos_20_filter.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC %md
# MAGIC 
# MAGIC Looks like I was right about positions >= 20 all being NA. But why are there 
# MAGIC NAs in positions 16-19 for some but not all entries? Looking at the tables
# MAGIC and the plot optinons above, we can ascertain two things:
# MAGIC 
# MAGIC 1. There are more missing values the lower the position (with 1 being the 
# MAGIC highest position).
# MAGIC 2. It can't purely be due to any driver-based attributes, since the same 
# MAGIC drivers have missing and non-missing milliseconds values even at positions
# MAGIC 16 and greater. It also can't be due to the time attribute, since all entries
# MAGIC with missing time also have missing milliseconds (but I don't think that's
# MAGIC what we're looking for). 
# MAGIC 
# MAGIC That leaves race, lap and position. Looking at them below.

# COMMAND ----------

df_milliseconds_na.groupby('raceId').agg(F.max('position')).count()

# COMMAND ----------

df_milliseconds_filled_pos_filter.groupby('raceId').agg(F.max('position')).count()

# COMMAND ----------

df_missing.groupby('raceId').agg(F.max('position')).count()

# COMMAND ----------

df_missing.groupby('lap').agg(F.max('position')).count()

# COMMAND ----------

df_milliseconds_na.groupby('lap').agg(F.max('position')).count()

# COMMAND ----------

df_milliseconds_race1_pos_16 = df_missing\
  .filter(F.col('position') >= 16)\
  .filter(F.col('raceId') == 1)
display(df_milliseconds_race1_pos_16)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC My suspicion was that the last 5 positions of each race were knocked off.
# MAGIC This explains why 20-24 were always empty, with more missing values the closer
# MAGIC a position is to 20.
# MAGIC 
# MAGIC I decided to look more closely at a specific race - race 1. This quickly 
# MAGIC disproved my hypothesis. Race 1 contains positions up to 19, but one laptime
# MAGIC for position 18 (Adrian Sutil), in lap 8, was recorded.
# MAGIC 
# MAGIC This eliminates a strict raceId + position based removal, because the same 
# MAGIC race has missing and non-missing values for the same driver, having the same
# MAGIC final position for the race.
# MAGIC 
# MAGIC A lap based explanation also seems implausible. The dataset
# MAGIC containing only missing milliseconds data has all but one of the laps 
# MAGIC represented. From the summary data, we can see that this is the maximum lap
# MAGIC number, lap 78. This is probably absent because people in the final positions
# MAGIC crash out and never get to that last lap, meaning the whole row is missing
# MAGIC instead of just the millisecond data. Incompletions are very common in F1.
# MAGIC So, the missingness pattern has little  to do with the lap number. 
# MAGIC 
# MAGIC The final possibility I can think of is lap + position, where earlier positions
# MAGIC have less laptimes taken off, and later positions have more laptimes taken off.
# MAGIC While this has already been established to be true from the bargraph of 
# MAGIC position and number of non-missing milliseconds data, I don't see a clear rule
# MAGIC to decide which laps are missed out. Race 1, Position 17 has missing values 
# MAGIC for lap 9, 13, 19, among others, whereas position 18 is missing laps 1-7, 
# MAGIC 9-21, 23 and 24.
# MAGIC 
# MAGIC It seems completely arbitrary to me. At this point, it seems like the pattern
# MAGIC of missing milliseconds data is weighted probabilistically according to position
# MAGIC from the range of 16-19. Milliseconds data is completely absent beyond 20, and 
# MAGIC all milliseconds data is available for positions above 15.
# MAGIC 
# MAGIC Moving on.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Nationality

# COMMAND ----------

df_nationality_na = df_missing.filter(F.col('nationality') == 'NA')
display(df_nationality_na)

# COMMAND ----------

display(df_nationality_na.summary())

# COMMAND ----------

# 47112 is the same number we saw in the groupby above, so we should be good

# COMMAND ----------

display(df_missing.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC Again this cannot be driver based - we see here a lot of missing values for
# MAGIC Lewis Hamilton, but above we saw some entries with Lewis Hamilton listed as
# MAGIC British. Since the same driver has both missing and non-missing values, a pure
# MAGIC driver-based criterion for missing data is ruled out.
# MAGIC 
# MAGIC That leaves raceId, lap, position, and timing data. I am not optimistic
# MAGIC here either, since the descriptives for these are almost identical to the 
# MAGIC whole df_missing dataset. This suggests that the df_nationality_na dataset
# MAGIC is randomly sampled from the full df_missing dataset, and hence that the
# MAGIC nationality missingness was randomly imposed.
# MAGIC 
# MAGIC Still I think I should have a look.

# COMMAND ----------

df_missing.groupby('raceId').agg(F.max('position')).count()

# COMMAND ----------

df_nationality_na.groupby('raceId').agg(F.max('position')).count()

# COMMAND ----------

df_missing.groupby('lap').agg(F.max('position')).count()

# COMMAND ----------

df_nationality_na.groupby('lap').agg(F.max('position')).count()

# COMMAND ----------

df_missing.groupby('position').agg(F.max('position')).count()

# COMMAND ----------

df_nationality_na.groupby('position').agg(F.max('position')).count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC All races, positions, and laps are represented in our nationality-missing 
# MAGIC dataset. There is also entries where nationality is missing for both 
# MAGIC valid and missing millisecond data.
# MAGIC 
# MAGIC These suggest that any "@ this position/lap/race, remove nationality"
# MAGIC isn't what is happening.
# MAGIC 
# MAGIC There could be a more complex hard and/or probabilistic rule "@ this
# MAGIC position, remove nationality for this lap/race/driver/timing range with
# MAGIC this probability". But that's going to be awfully difficult to see from data
# MAGIC exploration or even visualisation, especially given, the high number of laps
# MAGIC and races.
# MAGIC 
# MAGIC I think it's reasonable to assume that the nationality missingness is randomly
# MAGIC imposed, particularly given the lack of bias in the distributions of any other 
# MAGIC variable in the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Plot the distributions of variables with missing values against the observed ones and explain how different missingness patterns are affecting the distributions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prep data
# MAGIC 1. Get original data
# MAGIC 2. Use the fill_dummy_values function to prepare the data for the plot
# MAGIC 3. toPandas() - don't want to have to deal with matplotlib + pyspark

# COMMAND ----------

df_laptimes = spark.read.csv('/mnt/ne-gr5069/raw/lap_times.csv', header = True)
df_drivers = spark.read.csv('/mnt/ne-gr5069/raw/drivers.csv', header = True)

df_drivers_with_age = df_drivers\
  .withColumn("age", F.datediff(F.current_date(),F.col("dob"))/365.25)
df_drivers_with_age = df_drivers_with_age\
  .withColumn("age", df_drivers_with_age["age"].cast(IntegerType()))

df_original = df_laptimes.join(df_drivers_with_age, on = ['driverId'])
display(df_original.summary())

# COMMAND ----------

df_full = df_missing.join(df_original.select(F.col('driverId'),
                                             F.col('raceId'),
                                             F.col('lap'),
                                             F.col('nationality').alias('og_nat'),
                                             F.col('milliseconds').alias('og_ms'),
                                             F.col('age').alias('og_age')
                                            ), on = ['driverId', 'raceId', 'lap']
                         )

df_full = df_full.withColumn("og_ms", df_full["og_ms"].cast(IntegerType()))
display(df_full)

# COMMAND ----------

display(df_full.summary())

# COMMAND ----------

df_full_pd = df_full.toPandas()
df_full_dummy_vals = fill_dummy_values(df_full_pd, 0.075)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Age

# COMMAND ----------

#Get missing values for coloring
null_age = df_full_pd['age'].isnull()

#Generate Scatter plot
fig_age, ax = plt.subplots()
df_full_dummy_vals.plot(x = 'age',
                        y = 'og_age',
                        kind = 'scatter',
                        alpha = 0.5,
                        c = null_age,
                        cmap = 'rainbow',
                        figsize=(12,8),
                        grid = True,
                        legend = True,
                        ax = ax)
ax.legend(['Not Missing'])

# COMMAND ----------

fig_age_NA_compare = plt.figure(figsize = (12,10))
sns.distplot(df_full_pd['age'], hist = False, kde = True,
             color = 'red', 
             kde_kws={'linewidth': 2})
sns.distplot(df_full_pd['og_age'], hist = False, kde = True,
             color = 'blue', 
             kde_kws={'linewidth': 2})
display(fig_age_NA_compare)

# COMMAND ----------

display(fig_age)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC As a result of the missingness pattern in the age data, the overall
# MAGIC distribution of ages in df_missing is much younger. In the missing data,
# MAGIC there was no data above age 50.
# MAGIC 
# MAGIC There were some minor differences in how I calculated age from the
# MAGIC df_missing dataset (I used the current day's timestamp, subtracting
# MAGIC DOB). Nevertheless, for ages that were available, differences were 
# MAGIC minor (see scatterplot), suggesting this computational differences
# MAGIC would not be sufficient to account for significantly larger ages of
# MAGIC those whose ages were missing from df_missing.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Milliseconds

# COMMAND ----------

#Get missing values for coloring
null_ms = df_full_pd['milliseconds'].isnull()

#Generate Scatter plot
fig_ms, ax = plt.subplots()
df_full_dummy_vals.plot(x = 'milliseconds',
                        y = 'og_ms',
                        kind = 'scatter',
                        alpha = 0.5,
                        c = null_ms,
                        cmap = 'rainbow',
                        figsize=(12,8),
                        grid = True,
                        legend = True,
                        ax = ax)
ax.legend(['Not Missing'])

# COMMAND ----------

fig_ms_NA_compare = plt.figure(figsize = (12,10))
sns.distplot(df_full_pd['og_ms'], hist = False, kde = True,
             color = 'blue', 
             kde_kws={'linewidth': 2})
sns.distplot(df_full_pd['milliseconds'], hist = False, kde = True,
             color = 'red', 
             kde_kws={'linewidth': 2})
display(fig_ms_NA_compare)


# COMMAND ----------

display(fig_ms)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The overall distributions in laptimes seems very similar between the 
# MAGIC original observed data and the data contained in df_missing. In the 
# MAGIC density plot, the two curves overlap to the point where distinguishing 
# MAGIC them becomes difficult. The full data has slightly higher peaks, with
# MAGIC the most prominent at the first and highest peak near 0. I don't think
# MAGIC this is indicative of a skew in the df_missing laptimes data towards
# MAGIC faster laps, despite how missingness in the laptime data was related to
# MAGIC race position, as the higher blue peaks are true for all of the peaks,
# MAGIC and the scatterplot of original against df_missing data appears to have
# MAGIC a very similar distribution. I think it is more likely that the 
# MAGIC higher peaks reflect a greater frequency of data in the full dataset at
# MAGIC all points in the distribution, since it also includes data that was 
# MAGIC missing in the df_missing dataset.
# MAGIC 
# MAGIC The overall distribution of laptimes in df_missing thus does not seem
# MAGIC affected by the pattern of missingness in the df_missing dataset for
# MAGIC the milliseconds column.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nationality

# COMMAND ----------

nat_count = pd.DataFrame(df_full_pd['nationality'].value_counts())
nat_og_count = pd.DataFrame(df_full_pd['og_nat'].value_counts())
nat_count_df = nat_count.join(nat_og_count)

# COMMAND ----------

display(nat_count_df.plot(kind = 'bar'))

# COMMAND ----------

display(nat_count_df.plot.scatter(x = 'nationality',
                                  y = 'og_nat'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The overall frequency distribution of nationality in the full data and 
# MAGIC df_missing seem very similar. This is also seen in the perfectly linear
# MAGIC relationship between the two distributions in the scatterplot. 
# MAGIC 
# MAGIC Like with the laptimes data, the peaks
# MAGIC for each nationality is higher than the df_missing data, but that is likely 
# MAGIC due to the inclusion of entries that were missing in the df_missing dataset.
# MAGIC 
# MAGIC The similar shapes of the distributions suggests that the overall frequency
# MAGIC distribution of nationalities in df_missing was not affected by the pattern of
# MAGIC missingness in the df_missing dataset for the nationalities column. This also
# MAGIC somewhat supports the notion that the pattern of missingness in this column
# MAGIC was completely randomly imposed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Based on your findings, use at least two different imputation techniques to impute the variables for further analysis. Assess the effectiveness of imputation techniques both visually and analytically (e.g. using an appropriate accuracy metric that you defined).

# COMMAND ----------

# MAGIC %md
# MAGIC ### KNN imputation

# COMMAND ----------

df_missing_for_impute = df_missing.toPandas()
df_missing_for_impute = df_missing_for_impute[['driverId',
                                               'age',
                                               'raceId',
                                               'nationality',
                                               'lap',
                                               'position',
                                               'milliseconds']]

df_missing_for_impute = df_missing_for_impute.replace('NA', np.nan)

# COMMAND ----------

#Encode nationality as ordinal variable to run KNN imputer
encoder = OrdinalEncoder()
encoded_nat = encode(df_missing_for_impute['nationality'])
encoder.categories_

# COMMAND ----------

df_missing_for_impute['nationality'] = encoded_nat
df_missing_for_impute.head(15)

# COMMAND ----------

#Run KNNImputer
# This takes very long to run, so I've run it and saved the output in S3. 
# Will continue the assignment by reading this in.

#knn_imputer = KNNImputer(n_neighbors=10)
#fill missing values by imputing
#array_missing_imputed = knn_imputer.fit_transform(df_missing_for_impute)
#df_missing_imputed = pd.DataFrame(array_missing_imputed)
#spark_df_missing_imputed = spark.createDataFrame(df_missing_imputed)
#spark_df_missing_imputed.coalesce(1).write.csv('/mnt/xql2001-gr5069/interim/assignment_2/spark_df_missing_imputed.csv')

# COMMAND ----------

df_missing_imputed = spark.read.csv('/mnt/xql2001-gr5069/interim/assignment_2/spark_df_missing_imputed.csv', header = False)
display(df_missing_imputed)

# COMMAND ----------

pd_missing_imputed = df_missing_imputed.toPandas()
pd_missing_imputed = pd_missing_imputed.rename(columns={"_c0": "driverId",
                                                        "_c1": "age",
                                                        "_c2": "raceId",
                                                        "_c3": "nationality",
                                                        "_c4": "lap",
                                                        "_c5": "position",
                                                        "_c6": "milliseconds"}
                                              )
pd_missing_imputed = pd_missing_imputed.astype('float64', copy=False)
pd_missing_imputed['nationality'] = np.round(pd_missing_imputed['nationality'])

nat_dict = {0: 'American',
            1: 'Argentine',
            2: 'Australian',
            3: 'Austrian',
            4: 'Belgian',
            5: 'Brazilian',
            6: 'British',
            7: 'Canadian',
            8: 'Colombian',
            9: 'Czech',
            10: 'Danish',
            11: 'Dutch',
            12: 'Finnish',
            13: 'French',
            14: 'German',
            15: 'Hungarian',
            16: 'Indian',
            17: 'Indonesian',
            18: 'Irish',
            19: 'Italian',
            20: 'Japanese',
            21: 'Malaysian',
            22: 'Mexican',
            23: 'Monegasque',
            24: 'New Zealander',
            25: 'Polish',
            26: 'Portuguese',
            27: 'Russian',
            28: 'Spanish',
            29: 'Swedish',
            30: 'Swiss',
            31: 'Thai',
            32: 'Venezuelan'
           }

pd_missing_imputed['nationality'] = pd_missing_imputed['nationality'].map(nat_dict)
pd_missing_imputed.head(15)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Age

# COMMAND ----------

fig_age_imp_compare = plt.figure(figsize = (12,10))
sns.distplot(pd_missing_imputed['age'], hist = False, kde = True,
             color = 'red', 
             kde_kws={'linewidth': 2})
sns.distplot(df_full_pd['og_age'], hist = False, kde = True,
             color = 'blue', 
             kde_kws={'linewidth': 2})
display(fig_age_imp_compare)

# COMMAND ----------

missing_age_index = df_full_pd['age'].isna()
predicted_age_list = pd_missing_imputed['age'][missing_age_index]
actual_age_list = df_full_pd['og_age'][missing_age_index]
rmse_age = RMSE(predicted_age_list, actual_age_list)
rmse_age

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The KNN imputer was largely unsuccessful at imputing the missing values for 
# MAGIC age. The distribution of the data filled with the imputed values largely
# MAGIC resembles the distribution of ages in df_missing before imputation.
# MAGIC The distribution of ages remains younger than the actual data. One redeeming
# MAGIC factor of the KNN model is a higher peak at the high 40s in the imputed data
# MAGIC compared to the df_missing data, suggesting that the model successfully 
# MAGIC inferred that the drivers with missing age data were older. However, without
# MAGIC exposure to ages beyond 50, it was unable to impute the ages of these drivers
# MAGIC to be as old as they actually are.
# MAGIC 
# MAGIC Quantitatively, the limited effectiveness of the KNN imputer can be seen in
# MAGIC the RMSE value of 16.9, which is high for a distribution with a range of only
# MAGIC about 40.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Milliseconds

# COMMAND ----------

fig_ms_imp_compare = plt.figure(figsize = (12,10))
sns.distplot(df_full_pd['og_ms'], hist = False, kde = True,
             color = 'blue', 
             kde_kws={'linewidth': 2})
sns.distplot(pd_missing_imputed['milliseconds'], hist = False, kde = True,
             color = 'red', 
             kde_kws={'linewidth': 2})
display(fig_ms_imp_compare)

# COMMAND ----------

missing_ms_index = df_full_pd['milliseconds'].isna()
predicted_ms_list = pd_missing_imputed['milliseconds'][missing_age_index]
actual_ms_list = df_full_pd['og_ms'][missing_age_index]
rmse_ms = RMSE(predicted_ms_list, actual_ms_list)
rmse_ms

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC At first glance, the KNN imputer seemed largely successful in imputing 
# MAGIC missing laptime data. This is seen in the RMSE of 72906, which is small for a
# MAGIC  variable with a range in the millions. In the plot, we see that the 
# MAGIC imputation created a skew in laptimes towards faster laps, imputing more 
# MAGIC missing values as having lower values than was actually observed. This was 
# MAGIC likely due to the disproportionate volume of fast laps in the dataset. This
# MAGIC effect does not seem to be too severe given the small RMSE value and the fact 
# MAGIC that the imputed and actual data density plots are still very similar in 
# MAGIC shape.
# MAGIC 
# MAGIC Nevertheless, by creating the skew, the imputer created a laptime distribution
# MAGIC that is less similar to the distribution of the full data than the df_missing
# MAGIC laptime distribution. Rather than having a skew towards faster laps, however
# MAGIC minor, the df_missing laptime distribution was very similar to that of the full
# MAGIC data.
# MAGIC 
# MAGIC By shifting the distribution of missing data to be less similar to the full
# MAGIC data than the actual distribution, the KNN imputer proved to be less effective
# MAGIC than just removing missing values, at least for the case of representing the
# MAGIC distribution of F1 laptimes.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Nationality

# COMMAND ----------

fig_nat_imp_compare = plt.figure(figsize = (12,10))
sns.countplot(y = 'nationality',
              data=pd_missing_imputed,
              color = 'red',
              alpha = 0.5)
sns.countplot(y = 'og_nat',
              data=df_full_pd,
              color = 'blue',
              alpha = 0.5)
display(fig_nat_imp_compare)

# COMMAND ----------

missing_nat_index = df_full_pd['nationality'] == 'NA'
predicted_nat_list = pd_missing_imputed['nationality'][missing_nat_index]
actual_nat_list = df_full_pd['og_nat'][missing_nat_index]
confusion_matrix_nat = confusion_matrix(actual_nat_list, predicted_nat_list)
nat_accuracy = np.diagonal(confusion_matrix_nat).sum()/missing_nat_index.sum()
nat_accuracy

# COMMAND ----------

fig_nat_cm = plt.figure(figsize = (10,10))
sns.heatmap(confusion_matrix_nat)
display(fig_nat_cm)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The KNN imputer did a devastatingly poor job at imputing the missing 
# MAGIC nationalities data. This can be seen from:
# MAGIC 
# MAGIC 1. The dismal accuracy of 0.08 (though to be fair, chance is about 3%, so
# MAGIC the model is better than chance, but beating chance is very far from what
# MAGIC we need for missing values imputation).
# MAGIC 2. The density plot, where no nationality was at all well predicted, except
# MAGIC maybe for British and Germans.
# MAGIC 2. The heatmap of the confusion matrix (should have high counts on the 
# MAGIC diagonals, which is only somewhat true).
# MAGIC 
# MAGIC 
# MAGIC The imputer predicted less members of the frequent entries, like British 
# MAGIC and German, and more members of the infrequent nationalities such as Irish and
# MAGIC Danish. However, the model also tended to miss out other infrequent 
# MAGIC nationalities like Finnish and Brazillian.
# MAGIC 
# MAGIC This was likely because with the KNN imputer, we had to code nationality as
# MAGIC ordinal data, when it is in fact nominal. OneHotEncoding is more appropriate
# MAGIC for nominal data, but since it creates features based on membership to a
# MAGIC category, missing data would not appear to be missing under OneHotEncoding,
# MAGIC they would just appear as not a member of any class, leaving nothing to 
# MAGIC impute. The fact that driverId and raceId were also actually nominal features
# MAGIC but were included as continuous features were also likely contributors to the
# MAGIC issue.
# MAGIC 
# MAGIC *Honestly I didn't anticipate that this would be too much of a problem, and*
# MAGIC *have now run out of time to retrain new models for imputation, so I'm going*
# MAGIC *with what I have. In my defense, properly encoding driver and race Id probably*
# MAGIC *would have created a better model but also would've meant creating about 2000*
# MAGIC *new features. That would have made model training take a lot longer than I*
# MAGIC *have time for.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mean/Modal imputation

# COMMAND ----------

df_mean_mode_impute = df_full_pd.copy(deep = True)
df_mean_mode_impute['age'].fillna(df_mean_mode_impute['age'].mean(),
                                  inplace = True)
df_mean_mode_impute['milliseconds'].fillna(df_mean_mode_impute['milliseconds'].mean(),
                                           inplace = True)

mode_nationality = "German"
df_mean_mode_impute['nationality'] = df_mean_mode_impute['nationality'].str.replace('NA', mode_nationality)
df_mean_mode_impute.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Age

# COMMAND ----------

fig_age_mean_imp_compare = plt.figure(figsize = (12,10))
sns.distplot(df_mean_mode_impute['age'], hist = False, kde = True,
             color = 'red', 
             kde_kws={'linewidth': 2})
sns.distplot(df_full_pd['og_age'], hist = False, kde = True,
             color = 'blue', 
             kde_kws={'linewidth': 2})
display(fig_age_mean_imp_compare)

# COMMAND ----------

predicted_age_list_mean_imp = df_mean_mode_impute['age'][missing_age_index]
rmse_age_mean_imp = RMSE(predicted_age_list_mean_imp, actual_age_list)
rmse_age_mean_imp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The mean imputation was largely unsuccessful at imputing the missing values for 
# MAGIC age. Like the KNN imputer, the distribution of the data filled with the imputed
# MAGIC values largely resembles the distribution of ages in df_missing before
# MAGIC imputation, save for the tall peak at the mean age of about 37. Visually, the
# MAGIC mean imputation seems to have performed more poorly than KNN imputation given
# MAGIC the lower resemblance to the original distribution, due to the tall peak.
# MAGIC This is despite a slightly lower RMSE value for mean imputation  at 16.3 
# MAGIC instead of 16.9. Nevertheless, 16.2 remains high for a distribution with a 
# MAGIC range of only about 40. 
# MAGIC 
# MAGIC Like the KNN imputed distribution, the distribution of ages remains younger
# MAGIC than the actual data. Using only the mean, the mean imputation could not 
# MAGIC rightly infer that the drivers with missing ages were older, unlike the KNN
# MAGIC imputation.
# MAGIC 
# MAGIC Overall, the mean imputation was unsuccessful, and seemed to perform worse
# MAGIC than KNN imputation for filling in the missing age date.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Milliseconds

# COMMAND ----------

fig_ms_mean_imp_compare = plt.figure(figsize = (12,10))
sns.distplot(df_full_pd['og_ms'], hist = False, kde = True,
             color = 'blue', 
             kde_kws={'linewidth': 2})
sns.distplot(df_mean_mode_impute['milliseconds'], hist = False, kde = True,
             color = 'red', 
             kde_kws={'linewidth': 2})
display(fig_ms_mean_imp_compare)

# COMMAND ----------

predicted_ms_list_mean_imp = df_mean_mode_impute['milliseconds'][missing_age_index]
rmse_ms_mean_imp = RMSE(predicted_ms_list_mean_imp, actual_ms_list)
rmse_ms_mean_imp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Overall, the mean imputation seems successful at imputing missing values for
# MAGIC laptimes.
# MAGIC 
# MAGIC Quantitatively, the RMSE value of 8941 is very small compared to the range
# MAGIC of over 7 million. Visually, the mean imputed distribution lacks the skew
# MAGIC at around 500k milliseconds that was present in the KNN imputed distribution,
# MAGIC closely adhering to the original distribution. 
# MAGIC 
# MAGIC However, there are two notable deviations. 
# MAGIC 
# MAGIC First, the prominent blue peak at about 100000ms. This is more pronounced in 
# MAGIC this plot than in the plot from question 2 comparing the df_missing laptime
# MAGIC data to the full data, despite identical settings. This is very odd given
# MAGIC that mean imputation should have increased, not decreased the number of values 
# MAGIC at 100000ms, which is about the mean value that was imputed (about 94k).
# MAGIC After spending quite some time checking if my code was correct, I think it
# MAGIC probably was, and what we see is probably related to the smoothing of the plot
# MAGIC rather than changes in the distribution.
# MAGIC 
# MAGIC Second, blue peaks popping out over the red line at all peaks across the 
# MAGIC distributionshows that the mean imputation was imperfect at restoring the data
# MAGIC to its original state, with less red values across the range being similar to 
# MAGIC the plot from question 2 comparing the df_missing laptime data to the full
# MAGIC data.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Nationality

# COMMAND ----------

fig_nat_mode_imp_compare = plt.figure(figsize = (12,10))
sns.countplot(y = 'nationality',
              data=df_mean_mode_impute,
              color = 'red',
              alpha = 0.5)
sns.countplot(y = 'og_nat',
              data=df_full_pd,
              color = 'blue',
              alpha = 0.5)
display(fig_nat_mode_imp_compare)

# COMMAND ----------

predicted_nat_mode_imp_list = df_mean_mode_impute['nationality'][missing_nat_index]
confusion_matrix_mode_imp_nat = confusion_matrix(actual_nat_list, predicted_nat_mode_imp_list)
nat_mode_imp_accuracy = (np.diagonal(confusion_matrix_mode_imp_nat).sum())/missing_nat_index.sum()
nat_mode_imp_accuracy

# COMMAND ----------

fig_nat_mode_imp_cm = plt.figure(figsize = (10,10))
sns.heatmap(confusion_matrix_mode_imp_nat)
display(fig_nat_mode_imp_cm)

# COMMAND ----------

# MAGIC %md
# MAGIC The modal imputer did a poor job at imputing the missing nationalities data.
# MAGIC This can be seen from:
# MAGIC 
# MAGIC 1. The accuracy of 0.17, which is much better than chance at 0.03125, but far
# MAGIC from what we need as an imputer. That being said, modal imputation outperformed
# MAGIC the KNN imputer.
# MAGIC 2. The density plot, which predictably overestimated the modal nationality
# MAGIC (German) while leaving every other nationality lacking in representation
# MAGIC in the data.
# MAGIC 2. The heatmap of the confusion matrix (the colourful strip is German), and
# MAGIC the diagonal is pretty much black when it should be white.

# COMMAND ----------



# COMMAND ----------

#Unmount S3 buckets
#dbutils.fs.unmount("/mnt/%s" % MOUNT_NAME_RAW)
#dbutils.fs.unmount("/mnt/%s" % MOUNT_NAME_PROC)
