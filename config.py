# ViT-Base Configs (86M params)
ViT_BASE_CFG = {
    "L": 12, # layers
    "D": 768, # hidden size
    "MLP_SIZE": 3072,
    "N_HEADS": 12, # for MHA
    "QKV_BIAS": False, 
    "PATCH_SIZE": 16, #16x16
    "IMG_SIZE": 224, # 224x224
    "IN_CHANNELS": 3,
    "BATCH_SIZE": 64, 
    "p": 0.1, # dropout 
    "EPS": 10**-6,  # for layer norm
    "NUM_CLASSES": 1000,
    "LR": 10**-4,
    "BETA1": 0.9,
    "BETA2": 0.999,
    "ADAM_EPS": 10**-9,
    "EPOCHS": 7,
    "LOG_EVERY": 50,
} 

# WandB configs
entity = "rcadown8-fordham-university"
project = "ViT-Training"