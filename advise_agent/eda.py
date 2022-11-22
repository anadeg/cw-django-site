import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

TRAIN_FILE = "cw-train-2.csv"

data = pd.read_csv(TRAIN_FILE)

reg_nan = data["region"].isna().sum()

print(f'Region Data\nTotal Values \t|\t { data["region"].count() }\nMissing Values \t|\t {reg_nan}\n')

sns.countplot(x=data['region'])
plt.title('Region Distribution')
plt.show()

data['region'].fillna(data['region'].mode()[0], inplace=True)
data.fillna(0, inplace=True)

d_corr = data.corr()
fig, ax = plt.subplots()
ax = sns.heatmap(d_corr, cmap='YlGnBu')
plt.show()

# >= 0.3
# journ --- int_com, theo
# lingo --- teach, psy
# int_com --- journ, theo
# teach --- lingo, psy
# psy --- lingo, teach
# theo --- journ, int_com

# 0.2 <= ... < 0.3
# bio --- phys_cult
# geo --- hist, int_com, theo, physics
# design --- it
# journ --- ss
# it --- design, math
# hist --- geo, lingo, teach, ss, physics
# forest --- chem
# lingo --- journ, hist, ss
# math --- it, physics
# int_com --- geo,
# teach --- hist, physics
# psy --- law
# ss --- journ, hist, lingo, law
# theo --- geo
# physics --- geo, hist, math, teach
# turism --- phys_cult
# phys_cult --- bio, turism
# chem --- forest
# law --- psy, ss




