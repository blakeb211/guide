import pandas as pd
from scipy.stats import chi2_contingency
data = {
    '18–50': [15474, 104],
    '50–59': [5710, 145],
    '60–69': [4558, 285],
    '70–79': [2616, 399],
    '80–90': [1807, 363]
}

df = pd.DataFrame(data)
print(df)
print(f"def 4:",chi2_contingency(df.to_numpy()).statistic)
print(f"def 4:",chi2_contingency(df.to_numpy()).pvalue)
