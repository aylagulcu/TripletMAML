import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results= pd.read_csv("result_test.csv")
results = results.rename(columns=lambda x: x.strip()) # columns: ['test_er', 'test_acc']

metaval_accuracies = np.array(results['test_acc'])
mean = np.mean(metaval_accuracies, 0)
std = np.std(metaval_accuracies, 0)
ci95 = 1.96*std/np.sqrt(len(results))

print(f'Average accuracy: {mean} +/-{ci95}')
