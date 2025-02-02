import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#formula for EAD = Exposure + CCF *(unsulised limit)
#Exposure = current outstanding balanace that user has used
# unsulised limit = portion that hasnt been used
#CCF = % that estimates the likely hood that the UL will be used before default
# we will use a .75 ccf factor due to historical averages.

#example calculation 
#E = 60k
#UL = 40k
#CCF = .75

#EAD = 60,000 + (40,000 * .75) = 90,000
df = pd.read_csv("datasetNormalCCFEAD.csv")
# Mean value within 0.6 - 0.85 range
# Display the newly added columns
new_columns = ["CCF", "Undrawn_Amount", "EAD_With_Normal_CCF"]
df["EAD_With_Normal_CCF"] = df["EAD_With_Normal_CCF"].round(2)
print(df[new_columns].head())  # Print first 5 rows

