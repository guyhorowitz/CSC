params = {
    # learning params
    "batch_size": 64,
    "lr": 0.01,
    "epochs": 100,
    "tau": 4,
    "small_lamda": 0.1,
    "large_lamda": 1,
    "lamda_decay": 0.4,

    # samples params
    "time_steps": 10,
    "n_clean_train_samples": 1000,
    "n_samples_per_round": 200,
    "n_val_samples": 500,
    "n_test_samples": 2000
}