#!/usr/bin/env python3

import pandas as pd
import numpy as np
from Archeological_discovery import QuantumArchaeologicalOptimizer

print('🧪 Testing updated XGBoost model with positive R² optimization...')

try:
    # Initialize the optimizer
    optimizer = QuantumArchaeologicalOptimizer()

    print('\n🚀 TRAINING OPTIMIZED XGBOOST MODEL:')
    
    # Train optimized XGBoost model
    results = optimizer.train_xgboost_model_with_accuracy()

    # Write results to file
    with open('optimized_results.txt', 'w') as f:
        f.write('🎯 OPTIMIZED XGBOOST RESULTS\n')
        f.write('=' * 40 + '\n\n')
        
        f.write(f'R² Score: {results["test_metrics"]["r2"]:.6f}\n')
        f.write(f'RMSE: {results["test_metrics"]["rmse"]:.6f}\n')
        f.write(f'Correlation: {results["test_metrics"]["correlation"]:.6f}\n')
        
        if results["test_metrics"]["r2"] >= 0:
            f.write('\n🎉 SUCCESS: Achieved R² ≥ 0!\n')
            f.write('✅ Positive R² achieved with optimized approach\n')
        else:
            f.write(f'\n❌ Still negative R²: {results["test_metrics"]["r2"]:.6f}\n')
        
        f.write(f'\nData info:\n')
        f.write(f'  Features: {results["data_info"]["n_features"]}\n')
        f.write(f'  Samples: {results["data_info"]["n_samples"]}\n')

    print('✅ Results written to optimized_results.txt')
    print(f'✅ R² Score: {results["test_metrics"]["r2"]:.6f}')
    
    if results["test_metrics"]["r2"] >= 0:
        print('🎉 SUCCESS: Achieved positive R²!')
    else:
        print('❌ Still working on positive R²')

except Exception as e:
    with open('error_log.txt', 'w') as f:
        f.write(f'Error: {str(e)}\n')
    print(f'Error: {e}')
