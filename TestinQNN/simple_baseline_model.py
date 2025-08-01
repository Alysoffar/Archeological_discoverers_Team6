#!/usr/bin/env python3

# Simple approach to get R² ≥ 0
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

print("🎯 Simple approach to achieve R² ≥ 0...")
print("=" * 50)

# Load data
df = pd.read_csv('../dataset.csv')

# Target
y = df['AI Prediction Score'].values / 100.0

# Try different feature combinations to find what works
feature_sets = {
    'basic': ['Human Activity Index', 'Sonar Radar Detection'],
    'activity_focused': ['Human Activity Index', 'Sonar Radar Detection', 'Climate Change Impact'],
    'detection_focused': ['Sonar Radar Detection', 'Human Activity Index', 'Latitude', 'Longitude'],
    'comprehensive': ['Human Activity Index', 'Sonar Radar Detection', 'Climate Change Impact', 'Looting Risk (%)', 'Latitude', 'Longitude']
}

best_r2 = -np.inf
best_config = None

for name, features in feature_sets.items():
    print(f"\n🔍 Testing feature set: {name}")
    
    # Create feature matrix
    X = df[features].values
    
    # Handle any missing values
    X = np.nan_to_num(X, nan=np.nanmean(X))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Try different models
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost_simple': xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
        'XGBoost_reg': xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    }
    
    for model_name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            
            print(f"  {model_name:12s}: R²={r2:6.3f}, MSE={mse:.6f}, Corr={corr:.3f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_config = (name, model_name, model, scaler, features)
                
        except Exception as e:
            print(f"  {model_name:12s}: Error - {str(e)}")

print(f"\n🏆 BEST RESULT:")
print("=" * 30)
if best_config:
    feature_set_name, model_name, best_model, best_scaler, best_features = best_config
    print(f"Feature set: {feature_set_name}")
    print(f"Model: {model_name}")
    print(f"Best R²: {best_r2:.4f}")
    print(f"Features used: {best_features}")
    
    if best_r2 >= 0:
        print("✅ SUCCESS: Achieved R² ≥ 0!")
    else:
        print("⚠️ Still negative R², but this is the best we found")
        
    # Final test with the best configuration
    print(f"\n🔬 Final validation with best model...")
    X_final = df[best_features].values
    X_final = np.nan_to_num(X_final, nan=np.nanmean(X_final))
    X_final_scaled = best_scaler.transform(X_final)
    
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X_final_scaled, y, test_size=0.2, random_state=123  # Different random state
    )
    
    best_model.fit(X_train_final, y_train_final)
    y_pred_final = best_model.predict(X_test_final)
    
    final_r2 = r2_score(y_test_final, y_pred_final)
    final_mse = mean_squared_error(y_test_final, y_pred_final)
    final_corr = np.corrcoef(y_test_final, y_pred_final)[0, 1]
    
    print(f"Final validation R²: {final_r2:.4f}")
    print(f"Final validation MSE: {final_mse:.6f}")
    print(f"Final validation Corr: {final_corr:.4f}")
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        print(f"\nFeature importances:")
        for i, (feature, imp) in enumerate(zip(best_features, importances)):
            print(f"  {feature}: {imp:.4f}")
            
else:
    print("❌ No positive R² achieved with any configuration")

# If still no success, try a baseline approach
if best_r2 < 0:
    print(f"\n🔄 Trying baseline approach...")
    
    # Use only the most correlated single feature
    correlations = []
    feature_cols = ['Human Activity Index', 'Sonar Radar Detection', 'Climate Change Impact', 'Looting Risk (%)']
    
    for col in feature_cols:
        corr = np.corrcoef(df[col].values, y)[0, 1]
        correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    best_single_feature = correlations[0][0]
    
    print(f"Best single feature: {best_single_feature} (corr: {correlations[0][1]:.3f})")
    
    # Simple linear model with just this feature
    X_simple = df[best_single_feature].values.reshape(-1, 1)
    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
        X_simple, y, test_size=0.2, random_state=42
    )
    
    simple_model = LinearRegression()
    simple_model.fit(X_train_simple, y_train_simple)
    y_pred_simple = simple_model.predict(X_test_simple)
    
    simple_r2 = r2_score(y_test_simple, y_pred_simple)
    simple_corr = np.corrcoef(y_test_simple, y_pred_simple)[0, 1]
    
    print(f"Single feature R²: {simple_r2:.4f}")
    print(f"Single feature Corr: {simple_corr:.4f}")
    
    if simple_r2 >= 0:
        print("✅ SUCCESS with single feature approach!")
    else:
        print("❌ Even single feature approach failed")

print("\n🎯 Analysis completed!")
