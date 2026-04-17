import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(42)
n = 500

training = np.random.normal(3.8, 0.6, n).clip(1, 5)
psycap = (0.6 * training + np.random.normal(1.2, 0.4, n)).clip(1, 5)
tqm = (0.5 * training + 0.4 * psycap + np.random.normal(0.5, 0.3, n)).clip(1, 5)

df_final = pd.DataFrame({
    'Training_Needs': training,
    'PsyCap_Mediator': psycap,
    'TQM_Performance': tqm
})

X = sm.add_constant(df_final[['Training_Needs', 'PsyCap_Mediator']])
model = sm.OLS(df_final['TQM_Performance'], X).fit()

print("🚀 Hypothesis Testing Results for Tech-Line Industries:")
print(model.summary())

df_final.to_csv('TechLine_Final_Analysis.csv', index=False)