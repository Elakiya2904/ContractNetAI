from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

class AnalysisSummary(BaseModel):
    totalContracts: int
    linkedContracts: int
    sharedCounterparties: int
    avgRiskScore: float

class ContractPair(BaseModel):
    id: int
    contractA: str
    contractB: str
    sharedCounterparty: str
    riskLevel: str
    riskScore: float
    reasons: List[str]
    suggestions: List[str]
    transactionCount: int
    failureRate: float
    totalValue: int

class AnalysisResult(BaseModel):
    summary: AnalysisSummary
    contractPairs: List[ContractPair]

class StoredAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    fileName: str
    summary: dict
    contractPairs: List[dict]
    rawData: Optional[str] = None
