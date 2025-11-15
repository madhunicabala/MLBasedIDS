"""
IDS Model Inference UI with Demo Features
A Streamlit interface with flow generation and prediction demos
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# ==================== CONFIGURATION ====================
MODEL_DIR = "/content/drive/Shareddrives/CMPE279/CMPE279-IDSandAttackMitigation/saved_models/"

# Available models configuration
AVAILABLE_MODELS = {
    "DoS Attack Detection-RF": {
        "model": "dos_random_forest.joblib",
        "scaler": "dos_feature_scaler.joblib",
        "model_type": "sklearn",
        "description": "Random Forest model for DoS attack detection"
    },
    "DoS Attack Detection-XGBOOST": {
        "model": "dos_xgboost_model.json",
        "scaler": "dos_feature_scaler.joblib",
        "calibrator": "xgboost_calibrator.joblib",
        "model_type": "xgboost",
        "description": "XGBoost model for DoS attack detection (GPU accelerated)"
    },
    "DOS Detection-Baseline Model": {
        "model": "dos_logistic_regression.joblib",
        "scaler": "dos_feature_scaler.joblib",
        "model_type": "sklearn",
        "description": "Baseline Logistic Regression model for DoS detection"
    }
}
# ==================== SAMPLE DATA GENERATOR ====================
# ==================== SAMPLE DATA GENERATOR ====================
class NetworkFlowGenerator:
    """Generate network flows using REAL samples from test data"""
    
    def __init__(self):
        """Load real test samples from your saved data"""
        self.benign_samples = None
        self.attack_samples = None
        self.feature_names = None
        self._load_test_samples()
    
    def _load_test_samples(self):
        """Load pre-saved test samples"""
        try:
            # Try to load saved test samples
            test_data_path = Path(MODEL_DIR) / "test_samples.joblib"
            if test_data_path.exists():
                data = joblib.load(test_data_path)
                self.benign_samples = data['benign']
                self.attack_samples = data['attack']
                self.feature_names = data['feature_names']
                st.success(f"✓ Loaded {len(self.benign_samples)} benign & {len(self.attack_samples)} attack samples")
            else:
                st.warning("⚠️ Real test samples not found. Using synthetic data.")
                self._use_synthetic_fallback()
        except Exception as e:
            st.warning(f"⚠️ Could not load test samples: {e}. Using synthetic data.")
            self._use_synthetic_fallback()
    
    def _use_synthetic_fallback(self):
        """Fallback to synthetic data if real samples not available"""
        # Create dummy dataframes for synthetic mode
        self.benign_samples = pd.DataFrame()
        self.attack_samples = pd.DataFrame()
    
    def generate_benign_flow(self):
        """Pick a random benign sample from real test set"""
        if len(self.benign_samples) > 0:
            idx = np.random.randint(0, len(self.benign_samples))
            sample = self.benign_samples.iloc[idx]
            return pd.DataFrame([sample])
        else:
            # Synthetic fallback
            return self._generate_synthetic_benign()
    
    def generate_attack_flow(self, attack_type='dos'):
        """Pick a random attack sample from real test set"""
        if len(self.attack_samples) > 0:
            idx = np.random.randint(0, len(self.attack_samples))
            sample = self.attack_samples.iloc[idx]
            return pd.DataFrame([sample])
        else:
            # Synthetic fallback
            return self._generate_synthetic_attack()
    
    def generate_mixed_batch(self, n_flows=50, attack_ratio=0.3, attack_type='dos', n_features=77):
        """Generate mixed traffic from real samples"""
        flows = []
        labels = []
        
        for i in range(n_flows):
            if np.random.random() < attack_ratio:
                flows.append(self.generate_attack_flow())
                labels.append('Attack')
            else:
                flows.append(self.generate_benign_flow())
                labels.append('Benign')
        
        df = pd.concat(flows, ignore_index=True)
        df['True_Label'] = labels
        return df.sample(frac=1).reset_index(drop=True)
    
    def _generate_synthetic_benign(self):
        """Synthetic benign flow (fallback)"""
        flow = {
            'Flow Duration': np.random.randint(1000, 50000),
            'Total Fwd Packets': np.random.randint(5, 50),
            'Total Backward Packets': np.random.randint(5, 50),
            'Flow Bytes/s': np.random.uniform(1000, 100000),
            'Flow Packets/s': np.random.uniform(10, 500),
        }
        # Fill remaining features
        for i in range(72):
            flow[f'Feature_{i}'] = np.random.uniform(0, 10)
        return pd.DataFrame([flow])
    
    def _generate_synthetic_attack(self):
        """Synthetic attack flow (fallback)"""
        flow = {
            'Flow Duration': np.random.randint(100, 5000),
            'Total Fwd Packets': np.random.randint(100, 1000),
            'Total Backward Packets': np.random.randint(0, 10),
            'Flow Bytes/s': np.random.uniform(100000, 1000000),
            'Flow Packets/s': np.random.uniform(1000, 10000),
        }
        for i in range(72):
            flow[f'Feature_{i}'] = np.random.uniform(0, 50)
        return pd.DataFrame([flow])


# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model_components(model_name):
    """Load model and scaler"""
    components = {
        'model': None,
        'scaler': None,
        'calibrator': None,
        'model_name': model_name,
        'model_type': None
    }
    try:
        model_path = Path(MODEL_DIR)
        model_config = AVAILABLE_MODELS[model_name]
        components['model_type'] = model_config.get('model_type', 'sklearn')
        
        # Load model - different methods for XGBoost vs sklearn
        model_file = model_path / model_config["model"]
        if model_file.exists():
            if components['model_type'] == 'xgboost':
                import xgboost as xgb
                components['model'] = xgb.Booster()
                components['model'].load_model(str(model_file))
            else:
                components['model'] = joblib.load(model_file)
        
        # Load scaler
        scaler_file = model_path / model_config["scaler"]
        if scaler_file.exists():
            components['scaler'] = joblib.load(scaler_file)
        
        # Load calibrator (XGBoost only)
        if 'calibrator' in model_config:
            calibrator_file = model_path / model_config["calibrator"]
            if calibrator_file.exists():
                components['calibrator'] = joblib.load(calibrator_file)
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return components

# ==================== INFERENCE FUNCTION ====================
def predict(input_data, model_components):
    """Perform prediction"""
    try:
        # Remove label column if exists
        X = input_data.drop(columns=['True_Label'], errors='ignore')
        
        # Scale
        if model_components['scaler'] is not None:
            X_scaled = model_components['scaler'].transform(X)
        else:
            X_scaled = X
        
        # Predict based on model type
        if model_components['model'] is not None:
            if model_components['model_type'] == 'xgboost':
                import xgboost as xgb
                # Create DMatrix
                dmat = xgb.DMatrix(X_scaled)
                
                # Get raw probabilities
                probs_raw = model_components['model'].predict(dmat)
                
                # Apply calibration if available
                if model_components['calibrator'] is not None:
                    probs_calibrated = model_components['calibrator'].transform(probs_raw)
                    probs_calibrated = np.clip(probs_calibrated, 0, 1)
                else:
                    probs_calibrated = probs_raw
                
                # Create probability array for both classes
                probabilities = np.column_stack([1 - probs_calibrated, probs_calibrated])
                
                # Make predictions (default threshold 0.5)
                predictions = (probs_calibrated >= 0.5).astype(int)
                
            else:
                # sklearn models
                predictions = model_components['model'].predict(X_scaled)
                probabilities = model_components['model'].predict_proba(X_scaled)
            
            return predictions, probabilities
        
        return None, None
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(page_title="IDS Demo", layout="wide")
    st.title("🛡️ Network Intrusion Detection System - Demo")
    st.markdown("---")
    
    # Initialize generator ONCE per session
    if 'generator' not in st.session_state:
        st.session_state['generator'] = NetworkFlowGenerator()
    
    generator = st.session_state['generator']
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    selected_model = st.sidebar.selectbox(
        "Select Detection Model:",
        list(AVAILABLE_MODELS.keys()),
        index=0
    )
    
    st.sidebar.info(AVAILABLE_MODELS[selected_model]['description'])
    
    # Load model
    model_components = load_model_components(selected_model)
    
    # Model status
    st.sidebar.subheader("📊 Model Status")
    model_loaded = model_components['model'] is not None
    scaler_loaded = model_components['scaler'] is not None
    
    st.sidebar.write(f"Model: {'✅ Loaded' if model_loaded else '❌ Not loaded'}")
    st.sidebar.write(f"Scaler: {'✅ Loaded' if scaler_loaded else '❌ Not loaded'}")
    
    if not (model_loaded and scaler_loaded):
        st.error("⚠️ Model or scaler not loaded. Please check the model files.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Demo: Single Prediction", 
        "📦 Demo: Batch Prediction",
        "🔴 Demo: Live Simulation",
        "📁 Upload Your Data"
    ])
    
    # ==================== TAB 1: SINGLE PREDICTION DEMO ====================
    with tab1:
        st.header("Single Flow Prediction Demo")
        st.info("Generate synthetic network flows and test the model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🟢 Generate Benign Flow", use_container_width=True):
                st.session_state['demo_flow'] = generator.generate_benign_flow()
                st.session_state['demo_label'] = 'Benign'
                st.rerun()

        with col2:
                if st.button("🔴 Generate Attack Flow", use_container_width=True):
                    st.session_state['demo_flow'] = generator.generate_attack_flow()
                    st.session_state['demo_label'] = 'Attack'
                    st.rerun()
        
        if 'demo_flow' in st.session_state:
            st.subheader(f"Generated Flow: {st.session_state['demo_label']}")
            
            # Show key features
            df = st.session_state['demo_flow']
            #key_features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
            #              'Flow Bytes/s', 'Flow Packets/s']
            key_features = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol',
                'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 
                'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
                'Flow Byts/s', 'Flow Pkts/s', 
                'Fwd PSH Flags', 'Bwd PSH Flags', 'Idle Mean']
            display_features = {k: v for k, v in df.iloc[0].items() if k in key_features}
            st.json(display_features)
            
            # Predict
            if st.button("🔍 Analyze This Flow"):
                with st.spinner("Analyzing..."):
                    predictions, probabilities = predict(df, model_components)
                    
                    if predictions is not None:
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            pred_label = "🔴 Attack" if predictions[0] == 1 else "🟢 Benign"
                            st.metric("Prediction", pred_label)
                        
                        with col2:
                            confidence = probabilities[0][predictions[0]] * 100
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        with col3:
                            correct = (predictions[0] == 1 and st.session_state['demo_label'] == 'Attack') or \
                                     (predictions[0] == 0 and st.session_state['demo_label'] == 'Benign')
                            st.metric("Result", "✅ Correct" if correct else "❌ Wrong")
    
    # ==================== TAB 2: BATCH PREDICTION DEMO ====================
    with tab2:
        st.header("Batch Prediction Demo")
        st.info("Test the model on multiple flows at once")
        
        col1, col2 = st.columns(2)
        with col1:
            n_flows = st.slider("Number of flows", 10, 100, 50)
        with col2:
            attack_ratio = st.slider("Attack ratio", 0.0, 1.0, 0.35, 0.05)
        
        if st.button(" Generate & Analyze Batch", use_container_width=True):
            with st.spinner(f"Generating {n_flows} flows..."):
                # Generate data
                batch_df = generator.generate_mixed_batch(
                    n_flows=n_flows, 
                    attack_ratio=attack_ratio
                )
                
                # Predict
                predictions, probabilities = predict(batch_df, model_components)
                
                if predictions is not None:
                    # Add results
                    batch_df['Prediction'] = ['Attack' if p == 1 else 'Benign' for p in predictions]
                    batch_df['Confidence'] = [probabilities[i][predictions[i]] * 100 for i in range(len(predictions))]
                    batch_df['Correct'] = batch_df['True_Label'] == batch_df['Prediction']
                    
                    # Calculate metrics
                    accuracy = (batch_df['Correct'].sum() / len(batch_df)) * 100
                    tp = ((batch_df['Prediction'] == 'Attack') & (batch_df['True_Label'] == 'Attack')).sum()
                    fp = ((batch_df['Prediction'] == 'Attack') & (batch_df['True_Label'] == 'Benign')).sum()
                    tn = ((batch_df['Prediction'] == 'Benign') & (batch_df['True_Label'] == 'Benign')).sum()
                    fn = ((batch_df['Prediction'] == 'Benign') & (batch_df['True_Label'] == 'Attack')).sum()
                    
                    # Display metrics
                    st.success(" Batch Analysis Complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Flows", n_flows)
                    with col2:
                        st.metric("Accuracy", f"{accuracy:.1f}%")
                    with col3:
                        st.metric("True Positives", tp)
                    with col4:
                        st.metric("False Positives", fp)
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("True Negatives", tn)
                        st.metric("False Negatives", fn)
                    with col2:
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        st.metric("Precision", f"{precision:.2%}")
                        st.metric("Recall", f"{recall:.2%}")
                    
                    # Show results table
                    # Show results table
                    st.subheader("Prediction Results")

                    # Create display dataframe with key features
                    key_features = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 
                                    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
                                    'Flow Byts/s', 'Flow Pkts/s', 'Idle Mean']

                    # Prepare display columns
                    display_columns = ['True_Label', 'Prediction', 'Confidence', 'Correct'] + key_features

                    # Create the display dataframe
                    display_df = batch_df[display_columns].copy()

                    # Format the dataframe for better readability
                    display_df['Confidence'] = display_df['Confidence'].round(2)
                    display_df['Flow Duration'] = display_df['Flow Duration'].astype(int)
                    display_df['Tot Fwd Pkts'] = display_df['Tot Fwd Pkts'].astype(int)
                    display_df['Tot Bwd Pkts'] = display_df['Tot Bwd Pkts'].astype(int)
                    display_df['Flow Byts/s'] = display_df['Flow Byts/s'].round(2)
                    display_df['Flow Pkts/s'] = display_df['Flow Pkts/s'].round(2)
                    display_df['Idle Mean'] = display_df['Idle Mean'].round(2)

                    # Display with styling
                    st.dataframe(
                        display_df.head(20),
                        use_container_width=True,
                        height=400
                    )

                    # Add download button for full results
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label=" Download Full Results (CSV)",
                        data=csv,
                        file_name="batch_prediction_results.csv",
                        mime="text/csv"
                    )
                    #display_df = batch_df[['True_Label', 'Prediction', 'Confidence', 'Correct']].head(20)
                    #st.dataframe(display_df, use_container_width=True)
    
    # ==================== TAB 3: LIVE SIMULATION =================== #
    with tab3:
        st.header("Live Traffic Simulation")
        st.info("Simulate real-time network traffic monitoring")
        
        if 'simulation_running' not in st.session_state:
            st.session_state['simulation_running'] = False
        
        col1, col2 = st.columns(2)
        with col1:
            sim_duration = st.number_input("Flows to generate", 5, 50, 10)
        with col2:
            sim_delay = st.slider("Delay (seconds)", 0.1, 2.0, 0.5, 0.1)
        
        if st.button("▶️ Start Simulation", use_container_width=True):
            st.session_state['simulation_running'] = True
            
            # Placeholders for live updates
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            results_placeholder = st.empty()
            
            results = []
            attack_count = 0
            benign_count = 0
            
            for i in range(sim_duration):
                # Generate random flow
                is_attack = np.random.random() < 0.3
                if is_attack:
                    flow = generator.generate_attack_flow()
                    true_label = 'Attack'
                else:
                    flow = generator.generate_benign_flow()
                    true_label = 'Benign'
                
                # Predict
                predictions, probabilities = predict(flow, model_components)
                pred_label = 'Attack' if predictions[0] == 1 else 'Benign'
                confidence = probabilities[0][predictions[0]] * 100
                
                # Update counts
                if pred_label == 'Attack':
                    attack_count += 1
                else:
                    benign_count += 1
                
                # Store result
                results.append({
                    'Flow': i+1,
                    'True': true_label,
                    'Predicted': pred_label,
                    'Confidence': f"{confidence:.1f}%",
                    'Status': '✅' if true_label == pred_label else '❌'
                })
                
                # Update display
                status_placeholder.info(f"🔄 Processing flow {i+1}/{sim_duration}...")
                
                col1, col2, col3 = metrics_placeholder.columns(3)
                col1.metric("Processed", i+1)
                col2.metric("🟢 Benign", benign_count)
                col3.metric("🔴 Attacks", attack_count)
                
                # Show results table
                results_df = pd.DataFrame(results)
                results_placeholder.dataframe(results_df, use_container_width=True)
                
                time.sleep(sim_delay)
            
            status_placeholder.success("✅ Simulation Complete!")
    
    # ==================== TAB 4: UPLOAD YOUR DATA ====================
    with tab4:
        st.header("Upload Your Own Data")
        st.info("Upload a CSV file with network flow features")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                if st.button("🔍 Analyze Uploaded Data"):
                    with st.spinner("Analyzing..."):
                        predictions, probabilities = predict(df, model_components)
                        
                        if predictions is not None:
                            df['Prediction'] = ['Attack' if p == 1 else 'Benign' for p in predictions]
                            df['Confidence'] = [probabilities[i][predictions[i]] * 100 for i in range(len(predictions))]
                            
                            st.success("✅ Analysis Complete!")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total", len(predictions))
                            col2.metric("Benign", (predictions == 0).sum())
                            col3.metric("Attacks", (predictions == 1).sum())
                            
                            st.dataframe(df[['Prediction', 'Confidence']].head(20))
                            
                            # Download
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "predictions.csv",
                                "text/csv"
                            )
            
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()