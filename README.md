# ContractNetAI

**Cross-Contract Intelligence Platform for Smart Contract Relationship Analysis**

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Algorithms Used](#algorithms-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)

---

## 🎯 Problem Statement

In blockchain ecosystems, smart contracts frequently interact with each other and share common counterparties, creating complex dependency networks. These interconnected relationships pose several challenges:

1. **Hidden Dependencies**: Organizations struggle to identify which contracts are interdependent through shared counterparties
2. **Risk Concentration**: Multiple critical contracts depending on the same counterparty creates single points of failure
3. **Circular Dependencies**: Contracts with bidirectional relationships can create deadlock situations
4. **Transaction Failures**: High error rates in specific contract pairs remain undetected until critical failures occur
5. **Financial Exposure**: Lack of visibility into aggregate financial exposure across related contract pairs
6. **Manual Analysis Limitations**: Traditional approaches cannot efficiently analyze thousands of transaction records to detect patterns

**Impact**: Without automated cross-contract analysis, organizations face:
- Unexpected service disruptions when shared counterparties fail
- Unquantified financial risk from concentrated dependencies
- Difficulty in optimizing contract architecture
- Reactive rather than proactive risk management

---

## 💡 Proposed Solution

**ContractNetAI** provides an AI-powered platform that:

### Core Capabilities

1. **Automated Relationship Detection**
   - Analyzes transaction data to identify contracts sharing common counterparties
   - Aggregates duplicate relationships to provide unique contract pair insights
   - Visualizes dependency networks through directed graph analysis

2. **Multi-Dimensional Risk Scoring**
   - Combines heuristic analysis with Graph Neural Network (GNN) predictions
   - Evaluates 5 key risk factors: counterparty dependency, failure rates, financial exposure, circular dependencies, and transaction volume
   - Provides risk scores (0-100%) with categorization (Low/Medium/High)

3. **Contract-Specific Recommendations**
   - Generates unique, actionable recommendations for each contract pair
   - Tailors suggestions based on individual contract performance metrics
   - Identifies specific contracts requiring immediate attention

4. **Intelligent Filtering**
   - Displays top 5 highest-risk contract pairs for immediate action
   - Provides full dataset downloads for comprehensive analysis
   - Deduplicates reasons and recommendations for clarity

5. **Export & Reporting**
   - CSV export for spreadsheet analysis
   - TXT report with detailed explanations and recommendations

### Key Benefits

- **Proactive Risk Management**: Identify vulnerabilities before they cause failures
- **Data-Driven Decisions**: Quantified risk scores support prioritization
- **Cost Optimization**: Detect opportunities for transaction batching and gas reduction
- **Compliance Support**: Document dependencies for audit requirements
- **Scalability**: Processes thousands of transactions in seconds

---

## 📁 Project Structure

```
ContractNetAI/
├── backend/                          # FastAPI backend server
│   ├── analysis_engine.py           # Core analysis logic
│   ├── gnn_inference.py             # GraphSAGE inference
│   ├── models.py                    # Pydantic data models
│   ├── server.py                    # FastAPI application
│   ├── train_gnn.py                 # GNN training script
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Environment configuration
│   └── models/
│       └── graphsage_model.pth     # Trained GNN weights
│
├── frontend/                         # React frontend application
│   ├── public/
│   │   └── index.html               # HTML entry point
│   ├── src/
│   │   ├── App.js                   # Main application component
│   │   ├── index.js                 # React entry point
│   │   ├── mock.js                  # Mock data utilities
│   │   ├── components/
│   │   │   └── ui/                  # Reusable UI components (Radix UI)
│   │   ├── pages/
│   │   │   ├── Home.js              # Upload page
│   │   │   └── Results.js           # Analysis results display
│   │   ├── styles/
│   │   │   ├── Home.css
│   │   │   └── Results.css
│   │   └── lib/
│   │       └── utils.js             # Utility functions
│   ├── package.json                 # Node dependencies
│   ├── craco.config.js              # Create React App configuration
│   ├── tailwind.config.js           # Tailwind CSS configuration
│   └── .env                         # Frontend environment variables
│
├── tests/                            # Test directory
├── ethereum_transactions.csv         # Sample transaction data
├── sample_contracts.csv              # Additional sample data
└── README.md                         # This file
```

---

## 🛠 Tech Stack

### Backend

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core programming language | 3.13+ |
| **FastAPI** | RESTful API framework | 0.110.1 |
| **PyTorch** | Deep learning framework | 2.9.1 |
| **NetworkX** | Graph algorithms & analysis | 3.6.1 |
| **Pandas** | Data processing & manipulation | 2.3.3 |
| **NumPy** | Numerical computations | 2.3.5 |
| **Scikit-learn** | Machine learning utilities | 1.6.0 |
| **Uvicorn** | ASGI server | 0.25.0 |

### Frontend

| Technology | Purpose | Version |
|------------|---------|---------|
| **React** | UI framework | 19.0.0 |
| **React Router** | Client-side routing | 7.5.1 |
| **Axios** | HTTP client | 1.8.4 |
| **Tailwind CSS** | Utility-first CSS framework | 3.4.17 |
| **Radix UI** | Accessible component library | Various |
| **Lucide React** | Icon library | 0.507.0 |
| **CRACO** | Create React App configuration | 7.1.0 |

### Data Processing

- **CSV Parsing**: Pandas
- **Graph Construction**: NetworkX DiGraph (Directed Graph)
- **Feature Engineering**: Custom aggregation pipelines
- **Model Inference**: PyTorch with custom GraphSAGE implementation

---

## 🧠 Algorithms Used

### 1. **Graph Construction Algorithm**

**Purpose**: Build directed graph representation of contract interactions

**Algorithm**:
```
Input: Transaction DataFrame (from, to, value, is_error)
Output: Directed Graph G

For each transaction:
    If edge (from → to) exists:
        Update edge weights:
            - weight += 1 (transaction count)
            - value_sum += transaction value
            - fail_count += 1 if is_error
    Else:
        Create new edge with initial values
```

**Complexity**: O(T) where T = number of transactions

---

### 2. **Shared Counterparty Detection**

**Purpose**: Identify contract pairs with common dependencies

**Algorithm**:
```
Input: Transaction data
Output: Unique contract pairs with shared counterparties

1. Build user_contracts mapping: user → {contracts}
2. For each user with ≥2 contracts:
    For each combination of contract pairs (A, B):
        Aggregate pair_key = (A, B):
            Add user to sharedCounterparties set
3. Return deduplicated pairs
```

**Complexity**: O(U × C²) where U = users, C = avg contracts per user

---

### 3. **GraphSAGE (Graph Sample and Aggregate)**

**Purpose**: Learn node embeddings for risk prediction via Graph Neural Networks

**Architecture**:
```
Input Layer: 4 features (total_sent, total_received, out_degree, in_degree)
    ↓
GraphSAGE Layer 1: Aggregate neighbor features
    - Neighborhood aggregation via mean pooling
    - Concatenate self + aggregated features
    - Linear transformation: (in_feats * 2) → hidden_feats (64)
    - ReLU activation
    ↓
GraphSAGE Layer 2: Second aggregation
    - Similar aggregation process
    - Linear transformation: (hidden_feats * 2) → num_classes (2)
    - No activation (logits)
    ↓
Output: Risk classification (0: low risk, 1: high risk)
```

**Training**:
- Loss: Cross-Entropy
- Optimizer: Adam (lr=0.01, weight_decay=5e-4)
- Train/Test Split: 70/30
- Epochs: 30 (configurable)

**Complexity**: O(E × F × H) where E = edges, F = feature dim, H = hidden dim

---

### 4. **Multi-Factor Risk Scoring**

**Purpose**: Calculate comprehensive risk score for each contract pair

**Algorithm**:
```
Input: Contract pair (A, B), transaction stats, GNN scores (optional)
Output: Risk score (0.0-1.0), risk level, reasons

Initialize risk_score = 0.0

Factor 1: Shared Counterparty Dependency (weight: 0.3)
    If shared_count > 0:
        risk_score += 0.3

Factor 2: Transaction Failure Rate (weight: 0.25)
    avg_failure_rate = (A.failure_rate + B.failure_rate) / 2
    If avg_failure_rate > 10%:
        risk_score += 0.25
    Else if avg_failure_rate > 5%:
        risk_score += 0.15

Factor 3: Financial Exposure (weight: 0.25)
    total_value = A.value + B.value
    If total_value > $2M:
        risk_score += 0.25
    Else if total_value > $1M:
        risk_score += 0.15

Factor 4: Circular Dependencies (weight: 0.15)
    If path_exists(A→B) AND path_exists(B→A):
        risk_score += 0.15

Factor 5: Transaction Volume (weight: 0.05)
    If total_transactions > 300:
        risk_score += 0.05

Optional: GNN Augmentation
    If GNN scores available:
        gnn_avg = mean(GNN_score(A), GNN_score(B))
        risk_score = max(risk_score, gnn_avg)

Risk Level Classification:
    If risk_score ≥ 0.7: HIGH
    Else if risk_score ≥ 0.4: MEDIUM
    Else: LOW
```

**Complexity**: O(P × (S_A + S_B + G)) where P = pairs, S = stats computation, G = graph path search

---

### 5. **Recommendation Generation Algorithm**

**Purpose**: Generate contract-specific actionable recommendations

**Algorithm**:
```
Input: Risk level, contract stats (A, B), failure rates, value, circular flag
Output: List of unique recommendations (max 5)

Initialize suggestions = []

# Failure-based recommendations
If max(A.failure_rate, B.failure_rate) > 10%:
    Identify worst_contract
    Add: "Fix {rate}% error rate in {worst_contract}"
    Add: "Run diagnostics on {worst_contract}"

# Value-based recommendations
If total_value > $2M:
    Identify high_value_contract
    Add: "Implement transaction limits on {contract}"
    Add: "Enable real-time alerts for {counterparty} >$50k"

# Circular dependency recommendations
If has_circular:
    Add: "Break circular flow between A and B"
    Add: "Introduce intermediary contract"

# Volume-based recommendations
If transaction_count > 300:
    Add: "Batch transactions to reduce gas costs"

# Load balancing
If A.count > B.count * 2:
    Add: "Balance load: {A} handles {A.count} vs {B.count}"

# Risk-level specific
If risk_level == HIGH:
    Add: "Critical: Reduce dependency on {counterparty}"
Else if risk_level == MEDIUM:
    Add: "Diversify counterparty base"

Return unique(suggestions)[:5]
```

**Complexity**: O(1) per pair (fixed number of checks)

---

### 6. **Circular Dependency Detection**

**Purpose**: Identify bidirectional dependencies between contracts

**Algorithm**: Bidirectional Path Search using NetworkX
```
Input: Directed graph G, contracts A and B
Output: Boolean (circular dependency exists)

has_path_AB = BFS(G, source=A, target=B)
has_path_BA = BFS(G, source=B, target=A)

Return has_path_AB AND has_path_BA
```

**Complexity**: O(V + E) per check using BFS

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.13 or higher
- **Node.js**: 16 or higher
- **Yarn**: 1.22 or higher

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv ../.venv
../.venv/Scripts/activate  # Windows
source ../.venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the GNN model (optional, pre-trained weights included):
```bash
python train_gnn.py --csv ../ethereum_transactions.csv --epochs 30
```

5. Configure environment variables (create `.env` file):
```env
CORS_ORIGINS=*
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
yarn install
```

3. Configure environment variables (create `.env` file):
```env
REACT_APP_BACKEND_URL=http://localhost:8000/api
WDS_SOCKET_PORT=3000
REACT_APP_ENABLE_VISUAL_EDITS=false
ENABLE_HEALTH_CHECK=false
```

---

## 🎮 Usage

### Running the Application

**Terminal 1 - Backend Server:**
```bash
cd backend
../.venv/Scripts/python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend Development Server:**
```bash
cd frontend
yarn start
```

Access the application at `http://localhost:3000`

### Using the Platform

1. **Upload CSV**: Navigate to home page and upload transaction CSV file
   - Required columns: `from`, `to`, `value`
   - Optional column: `is_error` (or `error`)

2. **View Analysis**: Automatically redirected to results page showing:
   - Summary metrics (total contracts, linked contracts, shared counterparties, avg risk)
   - Top 5 highest-risk contract pairs with:
     - Risk score and level
     - Shared counterparty information
     - Transaction statistics
     - Unique "Why They Are Linked" reasons
     - Contract-specific recommendations

3. **Download Reports**:
   - **CSV**: Full data export for spreadsheet analysis
   - **TXT**: Comprehensive report with all pairs and recommendations

### Training Custom GNN Model

```bash
cd backend
python train_gnn.py --csv ../ethereum_transactions.csv --out models/graphsage_model.pth --epochs 50
```

Options:
- `--csv`: Path to training CSV file
- `--out`: Output path for model weights
- `--epochs`: Number of training epochs (default: 30)

---

## 📡 API Documentation

### Endpoints

#### `POST /api/analyze`

Upload CSV file and analyze contract relationships.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` (CSV file)

**Response**:
```json
{
  "summary": {
    "totalContracts": 100,
    "linkedContracts": 45,
    "sharedCounterparties": 12,
    "avgRiskScore": 0.52
  },
  "contractPairs": [
    {
      "id": 1,
      "contractA": "Supplier Agreement A",
      "contractB": "Vendor Contract B",
      "sharedCounterparty": "Gateway Contract C",
      "sharedCounterparties": ["0x123...", "0x456..."],
      "riskLevel": "high",
      "riskScore": 0.85,
      "reasons": [
        "2 shared counterparties: 0x123..., 0x456...",
        "High combined failure rate 12.5% across 0x789abc / 0xdef123",
        "High financial exposure $3,500,000 across pair",
        "Circular dependency pattern detected"
      ],
      "suggestions": [
        "Priority: Fix 15.2% error rate in Supplier Agreement A",
        "Implement transaction limits on Supplier Agreement A ($3,500,000 at risk)",
        "Break circular flow between Supplier Agreement A and Vendor Contract B",
        "Critical: Reduce dependency on Gateway Contract C"
      ],
      "transactionCount": 450,
      "failureRate": 12.5,
      "totalValue": 3500000
    }
  ]
}
```

#### `GET /api/`

Health check endpoint.

**Response**:
```json
{
  "message": "ContractNetAI API is running"
}
```

---

## 🔬 Model Performance

### GraphSAGE Training Results

- **Architecture**: 2-layer GraphSAGE with 64 hidden units
- **Training Data**: Ethereum transaction dataset
- **Final Accuracy**: 
  - Train: ~60.4%
  - Test: ~47.0%
- **Loss Convergence**: Achieved by epoch 30

### Risk Scoring Accuracy

- Multi-factor heuristic combined with GNN provides robust risk assessment
- Top 5 filtering ensures focus on highest-impact relationships
- Contract-specific recommendations provide actionable insights

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional GNN architectures (GAT, GCN)
- Temporal analysis of contract relationships
- Advanced visualization of dependency graphs
- Real-time monitoring integration
- Blockchain-specific optimizations

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Contact

For questions or support, please open an issue in the repository.

---

**Built with ❤️ using Python, React, and Graph Neural Networks**
