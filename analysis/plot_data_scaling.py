import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data_scaling.csv')

plt.plot(df['data_subset'], df['test_loss'])
plt.xlabel('Number of training cosmologies')
plt.ylabel('Test loss')

plt.savefig('data_scaling.png')