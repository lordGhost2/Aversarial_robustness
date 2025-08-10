import pandas as pd
import os

# Load individual reports
clean_df = pd.read_excel("results/clean_report.xlsx", index_col=0)
fgsm_df = pd.read_excel("results/fgsm_eps0.1_report.xlsx", index_col=0)
pgd_df = pd.read_excel("results/pgd_eps0.1_report.xlsx", index_col=0)

# Extract accuracy for summary
def extract_accuracy(df):
    return df.loc["accuracy"]["precision"] * 100  # sklearn saves accuracy under "precision" field

summary_df = pd.DataFrame({
    "Attack": ["Clean", "FGSM (ε=0.1)", "PGD (ε=0.1)"],
    "Accuracy (%)": [
        extract_accuracy(clean_df),
        extract_accuracy(fgsm_df),
        extract_accuracy(pgd_df)
    ]
})

# Write all to single Excel
with pd.ExcelWriter("results/Adversarial_Summary_Report.xlsx") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    clean_df.to_excel(writer, sheet_name="Clean Report")
    fgsm_df.to_excel(writer, sheet_name="FGSM (ε=0.1) Report")
    pgd_df.to_excel(writer, sheet_name="PGD (ε=0.1) Report")

print("✅ Combined Excel report saved as: results/Adversarial_Summary_Report.xlsx")
