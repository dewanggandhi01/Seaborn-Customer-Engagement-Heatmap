import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----- Style (professional) -----
sns.set_theme(style="white", context="talk")  # 'talk' for presentation-ready sizing

# ----- Synthetic, realistic customer engagement data -----
# We simulate customer-level engagement features with plausible relationships.
rng = np.random.default_rng(42)
n = 1200

# Latent engagement propensity (common factor)
engagement = rng.normal(loc=0.0, scale=1.0, size=n)

# Base signals + noise to induce correlations
site_visits       = np.clip(rng.normal(10 + 3*engagement, 3.0, n), 0, None)
session_duration  = np.clip(rng.normal(3.0 + 0.4*site_visits + 0.8*engagement, 2.0, n), 0.2, None)  # minutes
pages_per_visit   = np.clip(rng.normal(2.0 + 0.3*session_duration + 0.5*engagement, 1.0, n), 0.5, None)
email_opens       = np.clip(rng.normal(1.5 + 1.2*engagement + 0.1*site_visits, 1.2, n), 0, None)
click_through_rt  = np.clip(rng.normal(0.05 + 0.03*engagement + 0.005*pages_per_visit, 0.03, n), 0.0, 1.0)
app_sessions      = np.clip(rng.normal(5 + 1.5*engagement + 0.3*site_visits, 2.5, n), 0, None)
orders            = np.clip(rng.normal(0.6 + 0.05*engagement + 0.04*site_visits + 2.0*click_through_rt, 0.8, n), 0, None)
aov               = np.clip(rng.normal(45 + 2.0*pages_per_visit + 1.5*engagement, 10.0, n), 5, None)  # average order value
nps               = np.clip(rng.normal(6.0 + 0.6*engagement + 0.03*pages_per_visit + 0.04*orders, 1.8, n), 0, 10)  # 0–10
churn_risk        = np.clip(rng.normal(0.4 - 0.08*engagement - 0.02*site_visits - 0.05*nps, 0.15, n), 0.0, 1.0)   # 0–1

df = pd.DataFrame({
    "Site Visits": site_visits,
    "Session Duration (min)": session_duration,
    "Pages / Visit": pages_per_visit,
    "Email Opens": email_opens,
    "CTR": click_through_rt,
    "App Sessions": app_sessions,
    "Orders": orders,
    "AOV ($)": aov,
    "NPS (0-10)": nps,
    "Churn Risk": churn_risk,
})

# ----- Correlation matrix -----
corr = df.corr(numeric_only=True)

# ----- Heatmap -----
plt.figure(figsize=(8, 8))  # 8 in * 64 dpi = 512 px
ax = sns.heatmap(
    corr,
    cmap="mako",
    vmin=-1, vmax=1,
    annot=True, fmt=".2f",
    linewidths=0.5, linecolor="white",
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
)

plt.title("Customer Engagement Correlation Matrix — Retail Client", pad=14)
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)

# IMPORTANT: Save as exactly 512x512 (8 in at 64 dpi). Avoid bbox_inches='tight' to preserve exact size.
plt.savefig("chart.png", dpi=64)
plt.close()
