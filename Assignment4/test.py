import pysubdisc
import pandas as pd
import matplotlib.pyplot as plt

# Load the Adult data
data = pd.read_csv('adult.txt')

# Examine input data
table = pysubdisc.loadDataFrame(data)
print(table.describeColumns())

print('\n\n******* Section 1 *******\n')

# SECTION 1
# Set up SD with default settings, based on a 'single nominal' setting
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')

# Print the default settings
print(sd.describeSearchParameters())

# Do the actual run
sd.run()

# Print the subgroups
print(sd.asDataFrame())

print('\n\n******* Section 2 *******\n')

# SECTION 2
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'CORTANA_QUALITY'  # Use a valid quality measure
sd.qualityMeasureMinimum = 0.1  # Set a minimum quality threshold
sd.numericStrategy = 'NUMERIC_BEST'
# Run SD
sd.run(verbose=False)

# Print discovered subgroups
print(sd.asDataFrame())

print('\n\n******* Section 3 *******\n')

# SECTION 3
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'COVERAGE'  # Use a valid quality measure
sd.coverageMinimum = 10  # Set minimum coverage to 10 instances

# Run SD
sd.run(verbose=False)

# Print discovered subgroups
print(sd.asDataFrame())

print('\n\n******* Section 4 *******\n')

# SECTION 4
sd_no_filter = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd_no_filter.qualityMeasure = 'COVERAGE'  # Use a valid quality measure
sd_no_filter.filterSubgroups = False  # Disable filtering

# Run SD without filtering
sd_no_filter.run(verbose=False)

# Print discovered subgroups
print(sd_no_filter.asDataFrame())

# Compare subgroup counts
print("Subgroup count with filtering turned ON: ", len(sd.asDataFrame()))  # from Section 3
print("Subgroup count with filtering turned OFF: ", len(sd_no_filter.asDataFrame()))

# Compute pattern team of size 3: top 3 subgroups by quality score
df_no_filter = sd_no_filter.asDataFrame()
pattern_team = df_no_filter.sort_values(by='QualityMeasure', ascending=False).head(3)  # Replace 'QualityMeasure' with the correct column name
print("Pattern team of size 3:")
print(pattern_team)

print('\n\n******* Section 5 *******\n')

# SECTION 5
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'COVERAGE'  # Use a valid quality measure
sd.qualityMeasureMinimum = 0.0  # Set minimum quality to 0.0

# Run SD
sd.run(verbose=False)

# Print discovered subgroups
print(sd.asDataFrame())

print('\n\n******* Section 6 *******\n')

# SECTION 6
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'COVERAGE'  # Use a valid quality measure
sd.coverageMinimum = 5  # Minimum coverage set to 5 instances

# Run SD
sd.run(verbose=False)

# Print discovered subgroups
print(sd.asDataFrame())

print('\n\n******* Section 7 *******\n')

# SECTION 7
sd = pysubdisc.singleNumericTarget(data, 'age')  # Switch to numeric target (age)

sd.qualityMeasureMinimum = 0.0  # Set minimum quality to 0.0
sd.coverageMinimum = len(data) * 0.1  # Set coverage to 10% of the dataset

# Run SD
sd.run(verbose=False)

# Print average age and discovered subgroups
print("Average age in the data: ", data['age'].mean())
print(sd.asDataFrame())


print('\n\n******* Section 8 *******\n')

# SECTION 8
# Run 100 swap-randomised SD runs to determine the minimum required quality for significance level alpha = 0.05
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.swapRandomizedRuns = 100  # Set the number of swap-randomized runs

sd.run(verbose=False)

# Print minimum quality for significance
print("Minimum quality for significance: ", sd.qualityMeasureMinimum)
print(sd.asDataFrame())

print('\n\n******* Section 9 *******\n')

# SECTION 9
# Load the Ames Housing data
data_housing = pd.read_csv('ameshousing.txt')

# Examine input data
table_housing = pysubdisc.loadDataFrame(data_housing)
print(table_housing.describeColumns())

# Set up SD for Ames Housing data
sd_housing = pysubdisc.singleNumericTarget(data_housing, 'SalePrice')


# Run EMM
sd_housing.run(verbose=False)

# Print first discovered subgroup
print(sd_housing.asDataFrame().loc[0])

