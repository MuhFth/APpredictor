import joblib
import numpy as np
import pandas as pd

print("=" * 60)
print("DIAGNOSTIC CHECK - Model & Scaler")
print("=" * 60)

try:
    # Load model
    model = joblib.load("model_kelulusan.pkl")
    print("\n✅ Model loaded successfully")
    print(f"Model type: {type(model).__name__}")
    
    # Check if model has feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        print(f"\n📋 Model expects these features:")
        for i, name in enumerate(model.feature_names_in_, 1):
            print(f"  {i}. {name}")
        print(f"\nTotal features: {len(model.feature_names_in_)}")
    else:
        print("\n⚠️ Model doesn't have feature_names_in_ attribute")
        if hasattr(model, 'n_features_in_'):
            print(f"Number of features expected: {model.n_features_in_}")
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")

print("\n" + "=" * 60)

try:
    # Load scaler
    scaler = joblib.load("scaler_kelulusan.pkl")
    print("\n✅ Scaler loaded successfully")
    print(f"Scaler type: {type(scaler).__name__}")
    
    # Check scaler properties
    if hasattr(scaler, 'feature_names_in_'):
        print(f"\n📋 Scaler expects these features:")
        for i, name in enumerate(scaler.feature_names_in_, 1):
            print(f"  {i}. {name}")
        print(f"\nTotal features: {len(scaler.feature_names_in_)}")
    
    if hasattr(scaler, 'mean_'):
        print(f"\n📊 Scaler statistics:")
        print(f"Mean values: {scaler.mean_}")
        print(f"Scale values: {scaler.scale_}")
    
except Exception as e:
    print(f"\n❌ Error loading scaler: {e}")

print("\n" + "=" * 60)

# Test prediction with sample data
print("\n🧪 Testing with sample data...")
try:
    # Try with feature names
    test_data = pd.DataFrame({
        'Attendance (%)': [85.0],
        'Internal Test 1 (out of 40)': [30.0],
        'Internal Test 2 (out of 40)': [35.0],
        'Assignment Score (out of 10)': [8.0],
        'Final Exam Marks (out of 100)': [75.0]
    })
    
    print("\nTest input:")
    print(test_data)
    
    # Scale
    test_scaled = scaler.transform(test_data)
    print(f"\nScaled values: {test_scaled}")
    
    # Predict
    prediction = model.predict(test_scaled)[0]
    print(f"\n🎯 Prediction: {prediction:.2f}")
    
except Exception as e:
    print(f"\n❌ Error during prediction: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)