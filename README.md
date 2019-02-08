# Time-series-Recurrent-Neural-Network
This was done to the Keggle London meter data set (https://www.kaggle.com/jeanmidev/smart-meters-in-london).
The dataset is modified, in order to get the mean value for each day from each house for a given block.
In order to run the code:
numpy, matplotlib, pandas, keras, sklearn should be installed

in order to run the code dataset should be provided:
In the code: # load the dataset
give the file name and location to "read_csv('<file name and location>')"
When using for the normal datasets provided by Kaggle
set the "usecols=[2]" to "usecols=[3]"

Make sure that the dataset is ordered by the date before running since it is a time series
dataset.
