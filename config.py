# ViT-Base Configs (86M params)
ViT_BASE_CFG = {
    "L": 12, # layers
    "D": 768, # hidden size
    "MLP_SIZE": 3072,
    "NUM_HEADS": 12, # for MHA
    "PATCH_SIZE": 16, #16x16
    "IMG_SIZE": 224, # 224x224
    "IN_CHANNELS": 3,
    "BATCH_SIZE": 64, 
    "p": 0.1 # dropout 
} 

# WandB configs
entity = "rcadown8-fordham-university"
project = "ViT-Training"