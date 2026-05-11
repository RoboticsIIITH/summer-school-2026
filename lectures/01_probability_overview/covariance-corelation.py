# %%
import numpy as np
# raw data:         10.5, 5.4, 0.1, 1.6;
# 9.2, 4.3, -0.2, 1.2;
# 10.1, 5.1, 0.0, 1.5;
# 10.8, 5.6, 0.3, 1.8;
# 9.4, 4.6, -0.2, 1.4;
x = np.array(
    [
        [10.5, 5.4, 0.1, 1.6],
        [9.2, 4.3, -0.2, 1.2],
        [10.1, 5.1, 0.0, 1.5],
        [10.8, 5.6, 0.3, 1.8],
        [9.4, 4.6, -0.2, 1.4],
    ]
)
mean = np.mean(x, axis=0)
cov = np.cov(x, rowvar=False)
corr = np.corrcoef(x, rowvar=False)
print("Mean:\n", mean)
print("\nCovariance:\n", cov)
print("\nCorrelation:\n", corr)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(cov, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Covariance Matrix")
plt.show()
