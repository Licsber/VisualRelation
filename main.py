from src.models import base_model
import os


def main():
    config = {
        'num_batch': 128,
        'num_ways': 32,
        'num_shots': 32,
        'num_query': 1,
        'feature_dim': 32,
        'num_layers': 32,
        'in_feature': 32,
        'out_feature': 32,
        'class_num': 32,
    }
    
    base_model.train_test(config)

if __name__ == "__main__":
    main()
