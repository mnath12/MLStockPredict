#!/usr/bin/env python3
"""
Simple TensorFlow compatibility test for Mac M1/M2
"""

import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Architecture: {platform.architecture()}")

try:
    print("\n🔍 Testing TensorFlow import...")
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} imported successfully")
    
    print("\n🔍 Testing basic TensorFlow operations...")
    print(f"Keras available: {tf.keras.backend.is_keras_available()}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"CPU devices: {tf.config.list_physical_devices('CPU')}")
    
    print("\n🔍 Testing simple model creation...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    model = Sequential([
        Dense(10, input_shape=(1,)),
        Dense(1)
    ])
    print("✅ Model creation successful")
    
    print("\n🔍 Testing model compilation...")
    model.compile(optimizer='adam', loss='mse')
    print("✅ Model compilation successful")
    
    print("\n🔍 Testing simple training...")
    import numpy as np
    X = np.random.random((100, 1))
    y = np.random.random((100, 1))
    
    history = model.fit(X, y, epochs=1, verbose=0)
    print("✅ Simple training successful!")
    
    print(f"\n🎉 All TensorFlow tests passed!")
    print(f"   Final loss: {history.history['loss'][0]:.6f}")
    
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    print("\n💡 Solutions:")
    print("   1. Install TensorFlow: pip install tensorflow")
    print("   2. For Mac M1/M2, try: pip install tensorflow-macos")
    print("   3. Check Python version compatibility")
    
except Exception as e:
    print(f"❌ TensorFlow test failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    print("\n💡 This suggests a compatibility issue. Try:")
    print("   1. pip install --upgrade tensorflow")
    print("   2. For Mac M1/M2: pip install tensorflow-macos")
    print("   3. Check if you have conflicting packages") 