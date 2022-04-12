# %%
import cv2
import os
import random
import numpy as np
import pandas as pd

# %%
# define dir
data_dir = "../../crop_part1"
fnames_subset = os.listdir(data_dir)

# %%
# remove two files from the data set that do not follow the naming convention, therefore have not been labelled correctly and can't be used
del fnames_subset[8903]
del fnames_subset[6680]

# %% [markdown]
# ## Pre-Processing

# %%
# X: flattened version
# X_origin_dict: original version; key: index, value: 3-D np array
X_rgb = list()
X_origin_dict = dict()
for i, fname in enumerate(fnames_subset):
    # construct dir
    dir = data_dir + "/" + fname
    
    # read the data
    dat = cv2.imread(dir)

    dat = cv2.cvtColor(dat, cv2.COLOR_BGR2RGB)
    
    # store the original data
    X_origin_dict[i] = dat
    
    # store the data
    X_rgb.append(dat)

# convert to np array
#X_rgb1 = np.array(X_rgb)
#print("The shape of the X_rgb is:", X_rgb1.shape)

# %% [markdown]
# Combine data into a data frame

# %%
df = pd.DataFrame(columns=["image", "age", "race", "gender"])
age = []
race = []
gender =[]
for fname in fnames_subset:
    temp = fname.split("_")
    age.append(temp[0])
    race.append(temp[2])
    gender.append(temp[1])
    pass

# %%
# combine indian and asian labels into one "asian" label
race = ['2' if i=='3' else i for i in race]
# relabel "other" to 3
race = ['3' if i=='4' else i for i in race]

# %%
# assign lists into the dataframe
df.image = X_rgb
df.age = [int(i) for i in age]
df.race = [int(i) for i in race]
df.gender = [int(i) for i in gender]

# %%
df

# %%
df.shape

# %%
#Write data to csv for later usage
data_dir2 = "~/Documents/MIDS_Spring_semester/IDS705_Machinelearning/Final_team_project/team8_ML/20_data"
df.to_csv(f"{data_dir2}/image_data.csv")

# %% [markdown]
# ### LOOKING AT DISTRIBUTIONS OF AGE, RACE AND GENDER 
# #### This will help us understand the imbalance in the data

# %% [markdown]
# Labels
# 
# [age] is an integer from 0 to 116, indicating the age
# 
# [gender] is either 0 (male) or 1 (female)
# 
# [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern)

# %% [markdown]
# #### create a new dataset with labelled data

# %%
df1 = df.copy()

# %%
df1.head()

# %% [markdown]
# #### Race

# %%
#Writing a function that creates dummy variable for eacg race
def races(x):
        if x == 0 : return "white"
        if x == 1 : return "black"
        if x == 2 : return "asian"
        if x == 3 : return "others"

df1['race1'] = df1['race'].apply(races)

# %%
df1["race1"].value_counts()

# %% [markdown]
# #### Gender

# %%
#Writing a function that creates dummy variable for each gender
def genda(x):
        if x == 0 : return "male"
        if x == 1 : return "female"
df1['gender1'] = df1['gender'].apply(genda)

# %%
df1["gender1"].value_counts()

# %% [markdown]
# #### Age

# %%
#categorising age into age groups
df1.loc[(df1.age < 13),  'AgeGroup'] = 'Kid'
df1.loc[(df1.age >= 13) & (df1.age < 26 ),  'AgeGroup'] = 'Adolescents'
df1.loc[(df1.age >= 26) & (df1.age < 46 ),  'AgeGroup'] = 'Adults'
df1.loc[(df1.age >= 46),  'AgeGroup'] = 'mature'

# %%
df1["AgeGroup"].value_counts()

# %% [markdown]
# ### Visualising the classes for the different variables above

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# #### Racial Groupings

# %%
# Create a sample dataframe with an text index
race_d = pd.DataFrame(
    {"race1": [5265, 3005, 405, 1103]}, 
    index=["white", "asian", "black", "others"])
# Plot a bar chart
#race_d.plot(kind="bar")
from matplotlib import pyplot as plt
race_d['race1'].plot(kind="bar", title="Bar plot showing racial groupings", color ='maroon')
plt.xticks(rotation=0, horizontalalignment="center")
plt.title("Bar plot showing images by racial groupings")
plt.xlabel("Races")
plt.ylabel("Number of images")

# %% [markdown]
# #### Gender

# %%
# Create a sample dataframe with an text index
gender_d = pd.DataFrame(
    {"gender1": [5406, 4372]}, 
    index=["female", "male"])
# Plot a bar chart
#race_d.plot(kind="bar")
#from matplotlib import pyplot as plt
gender_d['gender1'].plot(kind="bar", title="Bar plot showing gender", color ='green')
plt.xticks(rotation=0, horizontalalignment="center")
plt.title("Bar plot showing images by gender")
plt.xlabel("gender")
plt.ylabel("Number of images")

# %% [markdown]
# #### Age groups

# %%
# Create a sample dataframe with an text index
age_d = pd.DataFrame(
    {"AgeGroup": [3255, 1726, 2115, 2682]}, 
    index=["<13yrs", "13to25yrs","26to45yrs", ">46yrs"])
# Plot a bar chart
#race_d.plot(kind="bar")
#from matplotlib import pyplot as plt
age_d['AgeGroup'].plot(kind="bar", title="Bar plot showing AgeGroups", color ='blue')
plt.xticks(rotation=0, horizontalalignment="center")
plt.title("Bar plot showing images by AgeGroups")
plt.xlabel("AgeGroups")
plt.ylabel("Number of images")


