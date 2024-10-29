import pysubdisc
import pandas
import matplotlib.pyplot as plt


# Load the Adult data
data = pandas.read_csv('adult.txt')

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
sd.qualityMeasure = 'CORTANA_QUALITY'
sd.qualityMeasureMinimum = 0.1

sd.numericStrategy = 'NUMERIC_BEST'

sd.run(verbose=False)

print(sd.asDataFrame())


print('\n\n******* Section 3 *******\n')

# SECTION 3
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'CORTANA_QUALITY'
sd.searchDepth = 2
sd.coverageMinimum = 10  # Set minimum coverage to 10 instances

sd.run(verbose=False)

print(sd.asDataFrame())



print('\n\n******* Section 4 *******\n')

# SECTION 4
sd_no_filter = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd_no_filter.qualityMeasure = 'CORTANA_QUALITY'
sd.searchDepth = 2
sd_no_filter.filterSubgroups = False  # Disable filtering

sd_no_filter.run(verbose=False)

print(sd_no_filter.asDataFrame())

print("Subgroup count with filtering turned ON: ", len(sd.asDataFrame()))	# reusing the result from Section 3 here
print("Subgroup count with filtering turned OFF: ", len(sd_no_filter.asDataFrame()))

# Compute pattern team of size 3 from the found subgroups
patternTeam, grouping = sd_no_filter.getPatternTeam(3,returnGrouping=True)


print('\n\n******* Section 5 *******\n')

# SECTION 5
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'COVERAGE'  # Use a valid quality measure
sd.qualityMeasureMinimum = 0.0  # Set minimum quality to 0.0
sd.searchDepth = 2

sd.run(verbose=False)

print(sd.asDataFrame())


print('\n\n******* Section 6 *******\n')

# SECTION 6
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasure = 'COVERAGE'  # Use a valid quality measure
sd.coverageMinimum = 5  # Minimum coverage set to 5 instances
sd.searchDepth = 2
sd.run(verbose=False)

print(sd.asDataFrame())


print('\n\n******* Section 7 *******\n')

# SECTION 7
sd = pysubdisc.singleNumericTarget(data, 'age')  # Switch to numeric target (age)
sd.searchDepth = 2
sd.qualityMeasureMinimum = 0.0  # Set minimum quality to 0.0
sd.coverageMinimum = len(data) * 0.1  # Set coverage to 10% of the dataset

sd.run(verbose=False)

print("Average age in the data: ", data['age'].mean())
print(sd.asDataFrame())


print('\n\n******* Section 8 *******\n')

# SECTION 8
# run 100 swap-randomised SD runs in order to determine the minimum required quality to reach a significance level alpha = 0.05
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.swapRandomizedRuns = 100  # Set the number of swap-randomized runs
sd.searchDepth = 2
sd.run(verbose=False)

print("Minimum quality for significance: ", sd.qualityMeasureMinimum)
print(sd.asDataFrame())


print('\n\n******* Section 9 *******\n')

# SECTION 9

# Load the Ames Housing data
data = pandas.read_csv('ameshousing.txt')

# Examine input data
table = pysubdisc.loadDataFrame(data)
print(table.describeColumns())

sd = pysubdisc.singleNumericTarget(data, 'SalePrice')

sd.run(verbose=False)

# Print first subgroup
print(sd.asDataFrame().loc[0])



