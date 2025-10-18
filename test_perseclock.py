from perseclock.main import run_perseclock

preds = run_perseclock(
    methylation_path="./example_data/example1_beta.csv.gz",
    pheno_path="./example_data/example1_pheno.csv",
    model_path="./perseclock/PerSE_model.pt",
    device="cpu",
    plot=True,
    save_path="./results/PerSEClock_Predicted_Ages.csv"
)