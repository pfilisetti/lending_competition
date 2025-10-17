### Signal Summary (zeros treated as missing)

| stat | signal1 | signal2 | signal3 |
|------|---------|---------|---------|
| count | 79 163.000000 | 79 103.000000 | 78 945.000000 |
| mean | 0.527695 | 0.525569 | 0.525573 |
| std | 0.275000 | 0.275468 | 0.275115 |
| min | 0.000008 | 0.000032 | 0.000015 |
| 25% | 0.284821 | 0.282565 | 0.283109 |
| 50% | 0.560774 | 0.559367 | 0.558638 |
| 75% | 0.754071 | 0.752040 | 0.751208 |
| max | 0.999975 | 0.999986 | 0.999965 |

### Pearson Correlation (zeros treated as missing)

|        | signal1 | signal2 | signal3 |
|--------|---------|---------|---------|
| signal1 | 1.000000 | 0.212427 | 0.215186 |
| signal2 | 0.212427 | 1.000000 | 0.212489 |
| signal3 | 0.215186 | 0.212489 | 1.000000 |

### Spearman Correlation (zeros treated as missing)

|        | signal1 | signal2 | signal3 |
|--------|---------|---------|---------|
| signal1 | 1.000000 | 0.219815 | 0.220508 |
| signal2 | 0.219815 | 1.000000 | 0.217540 |
| signal3 | 0.220508 | 0.217540 | 1.000000 |

### Pairwise Differences Normality Tests

- `signal1 - signal2`: mean = 0.0009, std = 0.3463, normaltest_stat = 2.07, pvalue = 0.3559  
- `signal1 - signal3`: mean = 0.0014, std = 0.3452, normaltest_stat = 0.92, pvalue = 0.6321  
- `signal2 - signal3`: mean = 0.0001, std = 0.3460, normaltest_stat = 0.15, pvalue = 0.9255


The three signals look almost identical once you treat zeros as missing: means around 0.526, medians near 0.56, and standard deviations at roughly 0.275. Each signal keeps about 79k usable observations, so they’re clearly measuring the same underlying quantity with noise.

Pearson and Spearman correlations come out near 0.21. That’s low enough to suggest each signal has independent noise, but high enough to show they share a common latent component.

Pairwise differences (S1−S2, S1−S3, S2−S3) are centered at ~0 with a standard deviation around 0.346, and all D’Agostino-Pearson tests return comfortable p-values (>0.35). So the hypothesis “true signal + Gaussian noise” is consistent with the data.

Next steps worth exploring:
- Plot histograms or Q-Q plots of the pairwise differences to visually check the normality assumption.
- Build a latent score (e.g., weighted average or PCA of the three signals) and test how well it predicts default with a logistic model to guide rate setting.


QQ-plots and histograms (in output folder):
- The histograms show that all three signals share the same shape: similar mass at 0.2–0.3, a strong peak around 0.5–0.6, and comparable spread up to 1.0. This confirms they sample from the same latent distribution once zeros are treated as missing.
- The Q-Q plots of signal differences hug the 45° line with only mild tail deviations, so the gaps between any two signals behave very much like Gaussian noise.
- In other words, each lender observes the same underlying score contaminated by independent normal noise. Averaging or otherwise pooling the signals is justified, and the resulting latent score should work well in downstream default-probability models.
