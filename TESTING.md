# Testing Guide for Quantum Archaeological Discovery System

## Test Files Overview

This document provides comprehensive information about the testing framework for the Quantum Archaeological Discovery System.

## Test Files

### 1. `Archeological_discovery.py` (Main System)
**Purpose**: Complete quantum archaeological analysis system
**Runtime**: ~2-3 minutes with optimized settings
**Usage**: `python Archeological_discovery.py`

**What it does**:
- Generates 100 sample Egyptian archaeological sites
- Trains quantum neural network (10 epochs)
- Performs site selection optimization
- Runs comprehensive analysis (clustering, temporal, success prediction)
- Provides detailed accuracy metrics and reporting

### 2. `test_optimization.py` (Quick Test)
**Purpose**: Fast optimization testing with minimal overhead
**Runtime**: ~30 seconds
**Usage**: `python test_optimization.py`

**What it does**:
- Generates 100 sample sites
- Quick QNN training (5 epochs)
- Optimizes 5 sites with budget/time constraints
- Displays site information with AI scores and archaeological metadata
- Shows success status based on prediction scores

### 3. `test_post_training.py` (Debug Suite)
**Purpose**: Component-wise testing and debugging
**Runtime**: ~1-2 minutes
**Usage**: `python test_post_training.py`

**What it does**:
- Tests quantum feature analysis in isolation
- Tests optimization module separately
- Tests analysis components individually
- Tests final reporting with safe error handling
- Provides detailed debugging information

## Test Execution Guide

### Quick Start Testing
```bash
# 1. Quick functionality check (30 seconds)
python test_optimization.py

# 2. Full system test (2-3 minutes)
python Archeological_discovery.py

# 3. Debug if issues occur
python test_post_training.py
```

### Expected Outputs

#### test_optimization.py Output Example:
```
🚀 Testing Site Optimization Display...
🧠 Quick QNN training...
⚛️ Running optimization...

🌟 Optimized Site Selection Results:
- EGY-123 | Score: 87/100 | Reino Nuevo | (25.687, 32.640)
  Materials: Oro, Caliza | Status: ✓ High Potential
- EGY-045 | Score: 76/100 | Reino Antiguo | (29.977, 31.133)
  Materials: Bronce, Arenisca | Status: ✓ High Potential

📊 Optimization Summary:
Total Selected: 5 sites
Total Estimated Cost: $78.45M
Selection Accuracy: 0.8234
```

#### Archeological_discovery.py Output Example:
```
🚀 Starting QNN Training with Comprehensive Accuracy Evaluation...

✅ QNN Training Complete!
Final Training Loss: 0.234567
Test Accuracy: 0.7856
Test F1-Score: 0.7423

🌟 Optimized Site Selection Results:
[Detailed site information with scores and metadata]

📊 Comprehensive Quantum Analysis Results:
[Clustering, temporal, and success prediction results]

🎯 FINAL ACCURACY SCORECARD:
   Neural Network: 78.5%
   Optimization: 82.3%
   Prediction: 76.1%
   🏆 OVERALL: 76.5%
   ✅ System Status: Ready for Archaeological Deployment!
```

## Performance Benchmarks

### Optimized Settings (Current)
- **Epochs**: 10 (reduced from 150)
- **Sites**: 100 (balanced for performance)
- **Features**: 25 (optimized selection)
- **Optimization Iterations**: 50 (reduced from 200)

### Timing Benchmarks
- **QNN Training**: ~15-30 seconds (10 epochs)
- **Site Optimization**: ~30-45 seconds (50 iterations)
- **Analysis Suite**: ~15 seconds
- **Total Runtime**: ~60-90 seconds (test_optimization.py)

## Troubleshooting Tests

### Common Test Issues

#### 1. Import Errors
```bash
# Test imports
python -c "from Archeological_discovery import *; print('✅ All imports successful')"
```

#### 2. Dataset Issues
```bash
# Verify dataset exists and is readable
python -c "import pandas as pd; df = pd.read_csv('Dataset_Arqueologico_Egipto_Expandido.csv'); print(f'Dataset shape: {df.shape}, Columns: {len(df.columns)}')"
```

#### 3. Memory Issues
- Reduce site count: Change `n_sites=100` to `n_sites=50`
- Reduce epochs: Change `epochs=10` to `epochs=5`
- Reduce optimization sites: Change `max_sites=10` to `max_sites=5`

#### 4. KeyError in Quantum Features
The `test_post_training.py` specifically addresses this with safe error handling:
```python
# Safe access with fallback values
bias_value = neuron_data.get('bias_value', 'N/A')
qc_ratio = neuron_data.get('quantum_classical_ratio', 'N/A')
```

### Test Configuration Options

#### Speed vs Accuracy Trade-offs
```python
# Fast Testing (30 seconds)
epochs=5, n_sites=50, max_sites=5, iterations=25

# Balanced Testing (1-2 minutes) - Default
epochs=10, n_sites=100, max_sites=10, iterations=50

# Thorough Testing (5+ minutes)
epochs=20, n_sites=200, max_sites=15, iterations=100, cross_validate=True
```

## Test Data

### Generated Site Features
- **Geographic**: Latitude, Longitude around famous Egyptian regions
- **Historical**: Random assignment to Egyptian periods (Old Kingdom, New Kingdom, etc.)
- **Archaeological**: Significance, accessibility, preservation urgency
- **Practical**: Estimated cost, duration, terrain difficulty
- **Ground Truth**: Realistic success probability based on site characteristics

### CSV Dataset Features
- **Site Identification**: Site ID, coordinates
- **AI Predictions**: AI Prediction Score (0-100)
- **Historical Context**: Time Period, Script Detected
- **Materials**: Material Composition, Human Activity Index
- **Risk Factors**: Climate Impact, Looting Risk
- **Detection**: Sonar Radar Detection values

## Advanced Testing

### Custom Test Scenarios
```python
# Test with specific parameters
python -c "
from test_optimization import test_site_optimization
# Modify parameters in the function call
test_site_optimization()
"
```

### Memory Profiling
```python
# Monitor memory usage during testing
import psutil, os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
```

### Performance Profiling
```python
import time
start_time = time.time()
# Run test
elapsed_time = time.time() - start_time
print(f'Test completed in {elapsed_time:.1f} seconds')
```

## Integration Testing

### Full Pipeline Test
1. **Data Generation**: Create synthetic Egyptian sites
2. **Feature Engineering**: Extract 49 features, select top 25
3. **QNN Training**: Train 4-layer quantum neural network
4. **Optimization**: Select optimal sites with constraints
5. **Analysis**: Clustering, temporal analysis, success prediction
6. **Reporting**: Comprehensive accuracy and performance metrics

### Validation Checks
- ✅ Data shapes match between X and y arrays
- ✅ Feature scaling in [0, 1] range
- ✅ Class balance within 40-60% range
- ✅ Prediction values in reasonable ranges
- ✅ Optimization respects budget/time constraints
- ✅ Analysis produces meaningful clusters

## Continuous Integration

### Automated Test Suite
```bash
#!/bin/bash
# run_tests.sh
echo "🚀 Running Quantum Archaeological Test Suite..."

echo "📊 Quick optimization test..."
python test_optimization.py

echo "🔬 Debug suite test..."
python test_post_training.py

echo "🎯 Full system test..."
python Archeological_discovery.py

echo "✅ All tests completed!"
```

### Success Criteria
- All tests complete without errors
- Overall system performance > 60%
- QNN training converges (loss decreases)
- Optimization finds valid site combinations
- Analysis produces reasonable accuracy metrics

---

**🧪 Testing Framework for Quantum Archaeological Discovery System**
