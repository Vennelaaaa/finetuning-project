
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming similarity_matrix is computed
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap="YlGnBu")
plt.savefig('../results/similarity_heatmap.png')
plt.show()
