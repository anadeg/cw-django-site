import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

TRAIN_FILE = "cw-train-2.csv"

data = pd.read_csv(TRAIN_FILE)

reg_nan = data["region"].isna().sum()

print(f'Region Data\nTotal Values \t|\t { data["region"].count() }\nMissing Values \t|\t {reg_nan}\n')

# sns.countplot(x=data['region'])
# plt.title('Region Distribution')
# plt.show()

data['region'].fillna(data['region'].mode()[0], inplace=True)
data.fillna(0, inplace=True)

# d_corr = data.corr()
# d_corr = d_corr.where((d_corr < 0.3) & (d_corr >= 0.2))
# fig, ax = plt.subplots()
# ax = sns.heatmap(d_corr, cmap='YlGnBu')
# plt.show()

# print(d_corr.to_string())

# >= 0.3
# journ --- int_com, theo
# lingo --- teach, psy
# int_com --- journ, theo
# teach --- lingo, psy
# psy --- lingo, teach
# theo --- journ, int_com

# 0.2 <= ... < 0.3
# bio --- phys_cult
# geo --- hist
# design --- it
# journ --- ss, lingo
# it --- design, math
# hist --- geo, lingo, teach, ss
# forest --- chem
# lingo --- journ, hist, ss
# math --- it, physics
# int_com --- teo,
# teach --- hist, physics
# ss --- journ, hist, lingo, theo
# theo --- int_com, ss
# physics --- math, teach
# turism --- phys_cult
# phys_cult --- bio, turism
# chem --- forest

def features_engineering(x):
    x['journ-int_com'] = x['journ'] + x['int_com']
    x['lingo-teach-psy'] = x['lingo'] + x['teach'] + x['psy']

    x['it-math'] = x['it'] + x['math']
    x['it-design'] = x['it'] + x['design']
    x['math-physics'] = x['math'] + x['physics']
    x['phys_cult-tourism'] = x['phys_cult'] + x['turism']
    x['ss-lingo-hist-journ'] = x['ss'] + x['lingo'] + x['hist'] + x['journ']
    x['phys_cult-bio'] = x['phys_cult'] + x['bio']
    x['chem-forest'] = x['chem'] + x['forest']
    x['journ-ss-lingo'] = x['journ'] + x['ss'] + x['lingo']

    x.drop(columns=['journ', 'int_com',
                    'lingo', 'teach', 'psy',
                    'it', 'math',
                    'design',
                    'physics',
                    'phys_cult', 'turism',
                    'ss',
                    'hist',
                    'bio',
                    'chem', 'forest'],
           inplace=True)

features_engineering(data)

d_corr = data.corr()
# d_corr = d_corr.where((d_corr < 0.3) & (d_corr >= 0.2))
fig, ax = plt.subplots()
ax = sns.heatmap(d_corr, cmap='YlGnBu')
plt.show()




