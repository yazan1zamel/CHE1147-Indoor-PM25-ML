# 🏠🌿 What Can Machine Learning Tell Us About the Air Inside Our Homes?

**A CHE1147 research highlight | Chemical Data Science and Engineering · University of Toronto · Team 4 · Fall 2026**

> **The central finding:** For predictions just 10 minutes ahead, the previous indoor PM₂.₅ measurement was the most accurate predictor *on average*. More complex machine-learning models did not beat this simple baseline on mean absolute error, but LightGBM reduced larger prediction errors and performed better for the highest observed concentration category.

## 🌬️ Why indoor air quality matters

We spend time indoors, but indoor air is not isolated from the outside world. Outdoor particles can enter buildings, while indoor activities, ventilation, and building conditions also affect particulate concentrations. Fine particulate matter, known as **PM₂.₅**, consists of airborne particles with diameters of approximately 2.5 micrometres or less. Exposure to fine particles is an established public-health concern, making the ability to understand changes in indoor concentrations a relevant research challenge [1].

Our project asks a focused question: **Can machine-learning models predict indoor PM₂.₅ over the next 10-minute interval more accurately than simply using the most recent measurement?**

## 🔬 From sensor readings to predictions

We analyzed a publicly documented residential monitoring study conducted by researchers associated with Lawrence Berkeley National Laboratory [2, 3]. The source study monitored **70 California homes** for approximately one week per home; **68 homes** provided usable observations for our prediction task.

The paper-aligned experiment included:

| Study component | Paper-aligned analysis |
|---|---:|
| Eligible households | **68** |
| Ten-minute prediction observations | **62,794** |
| Development / held-out households | **54 / 14** |
| Held-out observations | **12,104** |
| Historical and environmental predictors | **35** |
| Prediction horizon | **Next 10-minute interval** |

Predictors included preceding indoor and outdoor PM₂.₅ readings, ventilation and airflow indicators, temperature, humidity, cooking-related measurements, time variables, and building characteristics. Measurements from the target interval were not used as predictors. Model selection used household-grouped cross-validation, and the reported evaluation used households excluded from model fitting. The analysis compared **persistence, linear regression, ridge regression, LightGBM, and XGBoost**.

## 📊 The result was more interesting than a simple model ranking

| Prediction approach | Held-out MAE ↓ | Held-out RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| **Persistence** | **0.4173** | 2.6269 | 0.8712 |
| Linear regression | 0.5237 | 2.5918 | 0.8746 |
| Ridge regression | 0.5237 | 2.5918 | 0.8746 |
| **LightGBM** | 0.4832 | **2.4921** | **0.8840** |
| XGBoost | 0.4871 | 2.5373 | 0.8798 |

*MAE and RMSE are in µg/m³. Lower is better for both error metrics; higher is better for R². These figures describe the **Second Model / paper-aligned experiment**, not the repository's separate original exploratory workflow.*

![Comparison of held-out model errors](../assets/figures/model_comparison.png)

**Why did persistence win on MAE?** Indoor PM₂.₅ readings were highly correlated across consecutive 10-minute intervals. The most recent indoor measurement had a correlation of **r = 0.962** with the next interval's concentration in the development data. At such a short horizon, simply carrying forward the latest reading is a strong benchmark. The trained models could not improve on its *average absolute* error.

**Why did LightGBM win on RMSE?** RMSE penalizes large errors more strongly than MAE. LightGBM's RMSE was approximately **5.1% lower** than persistence's, even though its MAE was higher. This difference led us to look more closely at the relatively rare higher-concentration observations.

## 🌡️ Why elevated concentrations change the story

Of the 12,104 held-out observations, **8,561 (70.7%)** were below 5 µg/m³. Persistence performed best in that common low-concentration range.

Only **133 observations (1.1%)** were at or above 30 µg/m³. In this subset, LightGBM achieved an MAE of **11.871 µg/m³**, compared with **13.552 µg/m³** for persistence—an approximately **12.4% reduction**. XGBoost achieved **12.141 µg/m³**. These higher-concentration observations accounted for approximately **84.7% of persistence's total squared prediction error**, helping explain why RMSE and MAE favor different approaches.

![Model error by observed PM2.5 concentration range](../assets/figures/errors_by_concentration.png)

**The engineering lesson:** A model that is best for typical conditions may not be the model that minimizes the largest errors. The appropriate evaluation metric depends on what a monitoring application is intended to accomplish.

## 🧠 What the models learned—and what they did not

Both boosted-tree models relied strongly on previous indoor PM₂.₅ measurements. In XGBoost, the immediately preceding measurement represented approximately **85% of normalized gain-based feature importance**. This is a measure of how the *fitted model* uses predictors; it **does not** mean that one feature causes 85% of pollution, explains 85% of concentration variation, or contributes 85% to each prediction.

![Temporal autocorrelation in selected development homes](../assets/figures/temporal_autocorrelation.png)

Outdoor pollution, ventilation, and activity-related measurements also appeared among model features. Additional feature-ablation or controlled analyses would be required to establish their independent predictive value.

## 💡 Why these findings matter

This project offers three practical lessons for data-driven indoor air-quality research:

1. **Start with a meaningful baseline.** At short prediction horizons, a low-complexity method can be difficult to outperform. Any added computational complexity should demonstrate a measurable benefit.
2. **Evaluate the events that matter for the intended use.** Aggregate MAE can obscure large errors during uncommon elevated-concentration periods. Concentration-specific diagnostics reveal trade-offs that a single score misses.
3. **Build for generalization, not just fit.** Splitting data by household tests how methods transfer to different residences, which is a more demanding question than randomly mixing measurements from the same homes.

These findings may help inform future work on **indoor air-quality monitoring, event-focused forecasting, and the evaluation of smart-building prediction methods**. However, our study **did not test an operational alert system, demonstrate lower occupant exposure, or establish health benefits**. Those applications would require further testing.

## ⚖️ Limitations and next steps

The current task predicts only **one 10-minute interval ahead**. Concentrations were capped at **100 µg/m³** for modeling, potentially affecting the highest events. The original household monitoring periods were relatively short, and the held-out households had been examined in earlier project analyses; therefore, the reported evaluation is **not a completely fresh external validation**. Performance during abrupt concentration spikes still needs improvement.

The next research steps are to test **one-hour-ahead prediction**, remove the concentration cap in a sensitivity analysis, examine which features contribute incremental predictive value, evaluate peak-event detection, and validate on newly reserved households or monitoring periods.

---

## 📚 Sources and reproducibility

**External scientific sources**

1. World Health Organization. *WHO Global Air Quality Guidelines* (2021). https://www.who.int/publications/i/item/9789240034228
2. Singer, B. C., Chan, W. R., Kim, Y.-S., Offermann, F. J., & Walker, I. S. (2020). *Indoor air quality in California homes with code-required mechanical ventilation*. **Indoor Air**, 30(5), 885–899. https://doi.org/10.1111/ina.12676
3. Chan, W. R., Kim, Y.-S., Delp, W. W., Walker, I. S., & Singer, B. C. (2020). *Data from: Indoor air quality in California homes with code-required mechanical ventilation*. Dryad. https://doi.org/10.7941/D1ZS7X

**Project evidence**

- [Second Model: experiment description](../Second_Model/README.md)
- [Saved model metrics](../Second_Model/results/model_metrics.csv)
- [Executed Jupyter notebook](../Second_Model/Second_Model_Executed.ipynb)
- [Results dashboard (HTML file)](../Second_Model/results_dashboard.html) — download and open locally if GitHub does not render it.
- [Main project repository homepage](../README.md)

*All numerical results in this article come from the CHE1147 Second Model's saved experimental outputs. The article summarizes an academic course project; it does not present a deployed forecasting system.*
