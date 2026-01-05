# ContractNetAI - Integration Contracts

## API Endpoints

### 1. Upload CSV and Analyze
**Endpoint**: `POST /api/analyze`
**Request**: multipart/form-data with CSV file
**CSV Format**:
```
From,To,Value,TimeStamp,BlockHeight,TxHash,isError
0x123...,0x456...,1000000,1234567890,12345,0xabc...,0
```

**Response**:
```json
{
  "summary": {
    "totalContracts": 127,
    "linkedContracts": 43,
    "sharedCounterparties": 18,
    "avgRiskScore": 0.42
  },
  "contractPairs": [
    {
      "id": 1,
      "contractA": "Contract A",
      "contractB": "Contract B",
      "sharedCounterparty": "Counterparty X",
      "riskLevel": "high",
      "riskScore": 0.87,
      "reasons": ["Reason 1", "Reason 2"],
      "suggestions": ["Suggestion 1"],
      "transactionCount": 342,
      "failureRate": 12.4,
      "totalValue": 2300000
    }
  ]
}
```

### 2. Download CSV Report
**Endpoint**: `GET /api/download/csv/{analysis_id}`
**Response**: CSV file with contract relationships

### 3. Download TXT Report
**Endpoint**: `GET /api/download/txt/{analysis_id}`
**Response**: TXT file with detailed analysis report

## Backend Implementation Plan

### Phase 1: Core Analysis Engine
- ✅ CSV parsing and validation
- ✅ NetworkX graph construction
- ✅ Shared counterparty detection
- ✅ Circular dependency detection
- ✅ Risk scoring algorithm

### Phase 2: Risk Detection Logic

**Risk Scoring Factors**:
1. **Shared Counterparties** (weight: 0.3)
   - Multiple contracts depending on same entity
   
2. **Transaction Failure Rate** (weight: 0.25)
   - High isError percentage indicates unreliable contracts
   
3. **Financial Exposure** (weight: 0.25)
   - Total value concentration
   
4. **Circular Dependencies** (weight: 0.15)
   - Graph cycles indicate potential conflicts
   
5. **Transaction Frequency** (weight: 0.05)
   - Unusual spikes or patterns

**Risk Levels**:
- **High**: score >= 0.7
- **Medium**: 0.4 <= score < 0.7
- **Low**: score < 0.4

### Phase 3: Report Generation
- Generate human-readable explanations
- Create actionable suggestions
- Format CSV and TXT reports

## Frontend Integration Changes

### Replace Mock Data:
1. `Home.js`: 
   - Upload file to `/api/analyze`
   - Show loading state during analysis
   - Navigate to results with analysis data

2. `Results.js`:
   - Fetch analysis results from backend
   - Display real data instead of mock
   - Enable download buttons with actual API calls

### Files to Modify:
- `/app/frontend/src/pages/Home.js` - Add axios API call
- `/app/frontend/src/pages/Results.js` - Fetch real data
- Remove or keep `/app/frontend/src/mock.js` for demo mode

## Database Schema (MongoDB)

### Collection: `analyses`
```json
{
  "_id": "uuid",
  "timestamp": "2025-01-15T10:30:00Z",
  "fileName": "transactions.csv",
  "summary": { ... },
  "contractPairs": [ ... ],
  "rawData": "csv content"
}
```

## Python Dependencies
```
networkx>=3.0
pandas>=2.0
numpy>=1.24
```
