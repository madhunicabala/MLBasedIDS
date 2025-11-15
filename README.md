# DoS/DDoS Intrusion Detection System (IDS)

A sophisticated machine learning-based network intrusion detection system designed to identify and classify DoS/DDoS attacks in real-time using the CIC-IDS-2018 dataset.

## 🎯 Project Overview

This project implements a comprehensive intrusion detection system that leverages Random Forest and XGBoost algorithms to detect network attacks with over 99% accuracy. The system features multiple demonstration modes including single flow prediction, batch analysis, live traffic simulation, and a professional web interface for real-time monitoring.

## 🌟 Key Features

### Machine Learning Models
- **Random Forest Classifier**: Primary detection model with 99%+ accuracy
- **XGBoost Classifier**: Alternative high-performance model
- **Feature Engineering**: Utilizes 80+ network flow features from CIC-IDS-2018 dataset
- **Robust Preprocessing**: StandardScaler normalization for optimal performance

### Demonstration Modes
1. **Single Flow Prediction**: Generate and analyze individual network flows
2. **Batch Analysis**: Process multiple flows simultaneously with detailed metrics
3. **Live Traffic Simulation**: Real-time attack detection with animated monitoring
4. **Interactive Dashboard**: Professional web UI with comprehensive statistics
5. **REST API**: Programmatic access for integration with other systems

### Web Interface Features
- Real-time prediction with confidence scores
- Interactive flow generation (Benign/Attack)
- Batch processing with confusion matrix visualization
- Live traffic monitoring with attack/benign counters
- Downloadable results in CSV format
- Responsive design for presentations and demos

## 📊 Dataset

**CIC-IDS-2018 Dataset**
- Source: Canadian Institute for Cybersecurity
- Contains realistic network traffic with labeled attack types
- Features: 80+ flow-based statistical features
- Attack Types: DoS, DDoS, Brute Force, Web Attacks, Infiltration, Botnets

### Key Features Analyzed
- **Network Identity**: Source/Destination IP, Ports, Protocol
- **Traffic Volume**: Packet counts, Byte counts (Forward/Backward)
- **Flow Characteristics**: Duration, Bytes/s, Packets/s
- **Behavioral Indicators**: Flag counts, Idle times, Inter-arrival times
- **Statistical Measures**: Mean, Std, Min, Max of various metrics

## 🛠️ Technical Architecture

### Technology Stack
- **Language**: Python 3.8+
- **ML Libraries**: scikit-learn, XGBoost
- **Web Framework**: Streamlit (for UI), Flask (for API)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Google Colab, Cloudflare Tunnel

### System Components

```
┌─────────────────────────────────────────────────────┐
│                  User Interface                      │
│  (Streamlit Dashboard / Web Browser)                │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│              Web Server Layer                        │
│  • Streamlit Server (Port 8501)                     │
│  • Flask API Endpoints                              │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│           Prediction Engine                          │
│  • Model Loading & Management                       │
│  • Feature Scaling (StandardScaler)                 │
│  • Prediction & Probability Calculation             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│          Data Generation Layer                       │
│  • Real Sample Selection (from test set)            │
│  • Feature Extraction                               │
│  • Label Management                                 │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         ML Models & Data Storage                     │
│  • Random Forest Model (.pkl)                       │
│  • XGBoost Model (.pkl)                             │
│  • StandardScaler (.pkl)                            │
│  • Test Dataset (.csv)                              │
└─────────────────────────────────────────────────────┘
```

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/ids-dos-ddos-detection.git
cd ids-dos-ddos-detection
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download Required Files**
- Place the CIC-IDS-2018 dataset in the project directory
- Ensure you have the pre-trained models:
  - `rf_model_final.pkl`
  - `xgboost_model_final.pkl`
  - `scaler_final.pkl`

4. **Run the Application**
```bash
# For Streamlit UI
streamlit run IDS_UI_2_0.py --server.port 8501

# For Google Colab with Cloudflare Tunnel
# Run the notebook cells to start both Streamlit and tunnel
```

### Google Colab Setup

1. Upload the notebook to Google Colab
2. Mount your Google Drive (if models are stored there)
3. Run the setup cells to install dependencies
4. Execute the Streamlit launch cell
5. Access via the generated Cloudflare URL

## 📖 Usage Guide

### Single Flow Prediction

1. Navigate to the "Single Flow Prediction Demo" tab
2. Click "Generate Benign Flow" or "Generate Attack Flow"
3. Review the displayed features:
   - Flow Duration
   - Packet Counts (Forward/Backward)
   - Data Transfer Volumes
   - Flow Rates (Bytes/s, Packets/s)
   - Timing Characteristics
4. Click "Analyze This Flow" to get prediction
5. View results with confidence percentage

### Batch Analysis

1. Go to "Batch Prediction Demo" tab
2. Select number of flows (10-100)
3. Adjust attack ratio (0-100%)
4. Click "Generate & Analyze Batch"
5. Review comprehensive metrics:
   - Overall Accuracy
   - Confusion Matrix
   - Precision & Recall
   - Individual predictions with features
6. Download results as CSV for further analysis

### Live Traffic Simulation

1. Access "Live Traffic Demo" tab
2. Click "Start Monitoring"
3. Observe real-time flow processing
4. Monitor statistics:
   - Total flows processed
   - Attack detection count
   - Benign traffic count
   - Detection rate
5. Stop monitoring when complete

## 🎓 Understanding Attack Detection

### How the System Identifies Attacks

The model looks for specific patterns that distinguish malicious traffic:

#### Attack Indicators

1. **Extreme Packet Rates**
   - Normal: 10-50 packets/second
   - Attack: 400+ packets/second
   - **Example**: Flow with 420 pkts/s = Clear DoS attempt

2. **Unbalanced Communication**
   - Normal: Bidirectional traffic with responses
   - Attack: One-way traffic with no/minimal response
   - **Example**: 2 forward packets, 0 backward packets = Port scanning

3. **Amplification Patterns**
   - Attack: Small request → Large response
   - **Example**: 266 bytes sent, 935 bytes received = Reflection attack

4. **Zero Idle Time**
   - Normal: Natural pauses in human interaction
   - Attack: Continuous automated traffic
   - **Example**: Idle Mean = 0 = Bot/script activity

5. **Abnormal Flow Duration**
   - Attack: Very brief connections (< 50ms)
   - Normal: Longer, sustained connections
   - **Example**: 17ms duration with high rates = Hit-and-run attack

### Real Example from Demo

```json
{
  "Flow Duration": 16630,
  "Tot Fwd Pkts": 3,
  "Tot Bwd Pkts": 4,
  "TotLen Fwd Pkts": 266,
  "TotLen Bwd Pkts": 935,
  "Flow Byts/s": 72218.88,
  "Flow Pkts/s": 420.93,
  "Idle Mean": 0
}
```

**Classification: Attack (100% confidence)**

**Reasoning:**
- ⚠️ Packet rate of 420/s is 10-40x normal
- ⚠️ Amplification: 266 bytes in → 935 bytes out
- ⚠️ Zero idle time indicates automation
- ⚠️ Very brief duration (16ms)
- ⚠️ All indicators align = DDoS attack signature

## 📈 Model Performance

### Random Forest Model
- **Accuracy**: 99.0%+
- **Precision**: 93.33%
- **Recall**: 100%
- **F1-Score**: 96.5%

### XGBoost Model
- **Accuracy**: 99.0%+
- Similar performance metrics
- Faster inference time

### Validation Results
- **True Positives**: High detection of actual attacks
- **False Positives**: Minimal benign traffic misclassified
- **True Negatives**: Accurate benign traffic identification
- **False Negatives**: Near-zero missed attacks

## 🔬 Technical Details

### Feature Engineering

The system analyzes 80+ features including:

**Flow Identifiers**
- Source/Destination IP addresses
- Source/Destination ports
- Protocol type

**Volume Metrics**
- Total forward/backward packets
- Total forward/backward bytes
- Packet length statistics

**Rate Calculations**
- Flow Bytes/s
- Flow Packets/s
- Bulk rate metrics

**Behavioral Features**
- PSH/URG/FIN flag counts
- Idle time statistics
- Active time measurements
- Inter-arrival times

### Why Real Samples vs Synthetic Data?

Initial testing showed synthetic data generation problems:
- Artificially generated flows didn't match training distribution
- Models classified all synthetic data as benign
- **Solution**: Use actual samples from test dataset for demos
- This ensures realistic patterns and accurate predictions

### Model Training Process

1. **Data Loading**: Import CIC-IDS-2018 dataset
2. **Preprocessing**: Handle missing values, encode labels
3. **Feature Selection**: Use all relevant flow features
4. **Scaling**: StandardScaler normalization
5. **Training**: Random Forest with optimized hyperparameters
6. **Validation**: Cross-validation and test set evaluation
7. **Serialization**: Save models for deployment

## 🌐 Deployment Architecture

### Local Development
```
Localhost → Streamlit (Port 8501) → Browser
```

### Google Colab Deployment
```
Colab Container → Streamlit (8501) → cloudflared → Cloudflare Edge → Public URL
```

### Cloudflare Tunnel Explanation

**How It Works:**
1. Streamlit runs on internal port 8501 (private)
2. `cloudflared` creates outbound connection to Cloudflare
3. Cloudflare assigns public URL
4. Traffic flows: User → Cloudflare → Tunnel → Colab → Streamlit
5. Bypasses firewall restrictions (outbound connection allowed)

**Benefits:**
- No port forwarding needed
- Built-in DDoS protection
- Global CDN acceleration
- HTTPS encryption included
- Post-quantum cryptography (X25519MLKEM768)

## 🎤 Demo Presentation Tips

### For Live Demonstrations

1. **Start with Overview**: Explain DoS/DDoS attack context
2. **Show Single Flow**: Generate attack, explain features
3. **Highlight Key Indicators**: Point out suspicious patterns
4. **Run Batch Analysis**: Show scalability and metrics
5. **Live Monitoring**: Demonstrate real-time capabilities
6. **Q&A Preparation**: Be ready to explain any flow prediction

### Key Talking Points

- "Our system analyzes 80+ network features in real-time"
- "Notice the packet rate: 420/s is 10-40x normal traffic"
- "Zero idle time indicates automated attack tools"
- "The model achieved 99% accuracy on the CIC-IDS-2018 dataset"
- "We use actual test samples to ensure realistic demonstrations"

### Common Questions & Answers

**Q: Why only show 10-15 features?**
A: These are the most interpretable features for demonstrations. The model uses all 80+ features internally.

**Q: How does it handle new attack types?**
A: The model generalizes well to variations of DoS/DDoS patterns due to robust feature engineering.

**Q: Can it run on real network traffic?**
A: Yes, but would require integration with network monitoring tools (pcap capture, flow export).

**Q: What about false positives?**
A: Our system maintains <7% false positive rate with 93%+ precision.

## 🔒 Security Considerations

- Models are trained on labeled data only
- No live network access in current implementation
- Cloudflare tunnel provides encrypted connection
- No sensitive data stored in web interface
- Results are session-based (not persisted)

## 🛣️ Future Enhancements

### Planned Features
- [ ] Multi-class attack classification (DoS types)
- [ ] Real-time packet capture integration
- [ ] Automated model retraining pipeline
- [ ] Alert notification system
- [ ] Historical attack pattern analysis
- [ ] Integration with SIEM systems
- [ ] Mobile app for monitoring

### Research Directions
- Deep learning models (LSTM, CNN)
- Ensemble methods combining multiple models
- Adversarial attack resistance
- Zero-day attack detection
- Encrypted traffic analysis

## 📚 References

### Dataset
- Canadian Institute for Cybersecurity. (2018). *CIC-IDS-2018 Dataset*. 
  https://www.unb.ca/cic/datasets/ids-2018.html

### Research Papers
- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"

### Technologies
- **Scikit-learn**: Machine learning framework
- **XGBoost**: Gradient boosting library
- **Streamlit**: Web application framework
- **Cloudflare**: CDN and tunnel service

## 👥 Contributors

- **Madhunica** - Project Developer

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Canadian Institute for Cybersecurity for the CIC-IDS-2018 dataset
- Open-source community for ML libraries and frameworks
- Cloudflare for providing free tunnel service
- Google Colab for computational resources

## 📞 Contact & Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/madhunicabala/MLBasedIDS/issues)
- **Email**: madhunicabala.92@gmail.com
- **LinkedIn**: [madhunicabala]

## 🎯 Project Status

**Current Version**: 2.0
**Status**: Active Development
**Last Updated**: November 2025

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

## Quick Start Commands

```bash
# Clone repository
git clone https://github.com/madhunicabala/<<repo_name>>

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run IDS_UI_2_0.py

# Access using cloudflare URL
```

## Project Structure

```
ids-dos-ddos-detection/
├── IDS_UI_2_0.py                 # Streamlit web interface
├── IDS_279_Updated_ForDEMO.ipynb # Main notebook with training
├── requirements.txt              # Python dependencies
├── models/
│   ├── rf_model_final.pkl       # Random Forest model
│   ├── xgboost_model_final.pkl  # XGBoost model
│   └── scaler_final.pkl         # Feature scaler
├── data/
│   └── test_data.csv            # Test dataset samples
├── docs/
│   └── screenshots/             # Demo screenshots
└── README.md                     # This file
```

---

**Built with ❤️ for Network Security**
