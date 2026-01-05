// Mock data for ContractNetAI demo

export const mockAnalysisResults = {
  summary: {
    totalContracts: 127,
    linkedContracts: 43,
    sharedCounterparties: 18,
    avgRiskScore: 0.42
  },
  contractPairs: [
    {
      id: 1,
      contractA: "Supplier Agreement A",
      contractB: "Vendor Contract B",
      sharedCounterparty: "Gateway Services X",
      riskLevel: "high",
      riskScore: 0.87,
      reasons: [
        "Shared critical vendor dependency detected",
        "High transaction failure rate (12.4%)",
        "Concentrated financial exposure: $2.3M"
      ],
      suggestions: [
        "Review vendor obligation terms across both contracts",
        "Implement redundancy plan for Gateway Services X",
        "Monitor transaction success rates weekly",
        "Consider financial exposure limits or distribution"
      ],
      transactionCount: 342,
      failureRate: 12.4,
      totalValue: 2300000
    },
    {
      id: 2,
      contractA: "Partnership Agreement C",
      contractB: "Joint Venture D",
      sharedCounterparty: "Payment Processor Y",
      riskLevel: "medium",
      riskScore: 0.56,
      reasons: [
        "Circular dependency pattern identified",
        "Moderate transaction volume spike (45% increase)",
        "Shared payment processing infrastructure"
      ],
      suggestions: [
        "Audit circular transaction flows for inefficiencies",
        "Diversify payment processing channels",
        "Establish transaction volume thresholds and alerts",
        "Review dependency chain for optimization opportunities"
      ],
      transactionCount: 187,
      failureRate: 3.2,
      totalValue: 890000
    },
    {
      id: 3,
      contractA: "Service Level Agreement E",
      contractB: "Maintenance Contract F",
      sharedCounterparty: "Infrastructure Provider Z",
      riskLevel: "low",
      riskScore: 0.23,
      reasons: [
        "Shared infrastructure provider",
        "Low transaction failure rate (1.1%)",
        "Stable interaction patterns"
      ],
      suggestions: [
        "Continue monitoring shared infrastructure health",
        "Maintain current redundancy protocols",
        "Schedule periodic review of provider performance"
      ],
      transactionCount: 98,
      failureRate: 1.1,
      totalValue: 420000
    },
    {
      id: 4,
      contractA: "Licensing Agreement G",
      contractB: "Consultancy Contract H",
      sharedCounterparty: "Technical Services M",
      riskLevel: "medium",
      riskScore: 0.61,
      reasons: [
        "Multiple contract dependency on single service provider",
        "Unusual interaction frequency detected",
        "Moderate failure rate (6.8%)"
      ],
      suggestions: [
        "Evaluate alternative technical service providers",
        "Investigate cause of interaction frequency anomalies",
        "Implement service level monitoring dashboards",
        "Consider backup provider arrangements"
      ],
      transactionCount: 215,
      failureRate: 6.8,
      totalValue: 1150000
    },
    {
      id: 5,
      contractA: "Sales Agreement I",
      contractB: "Distribution Contract J",
      sharedCounterparty: "Logistics Partner N",
      riskLevel: "high",
      riskScore: 0.79,
      reasons: [
        "Critical logistics dependency across sales channels",
        "High value concentration ($3.1M)",
        "Recent transaction spike (67% increase)"
      ],
      suggestions: [
        "Diversify logistics partnerships immediately",
        "Establish contingency fulfillment plans",
        "Monitor transaction spike sustainability",
        "Review capacity planning with Logistics Partner N"
      ],
      transactionCount: 428,
      failureRate: 4.9,
      totalValue: 3100000
    }
  ]
};

export const generateCSVReport = (data) => {
  const header = "Contract A,Contract B,Shared Counterparty,Risk Level,Risk Score,Transaction Count,Failure Rate (%),Total Value ($)\n";
  const rows = data.contractPairs.map(pair => 
    `"${pair.contractA}","${pair.contractB}","${pair.sharedCounterparty}",${pair.riskLevel},${pair.riskScore},${pair.transactionCount},${pair.failureRate},${pair.totalValue}`
  ).join("\n");
  return header + rows;
};

export const generateTXTReport = (data) => {
  let report = "===================================================\n";
  report += "         CONTRACTNETAI ANALYSIS REPORT\n";
  report += "         Cross-Contract Intelligence Report\n";
  report += "===================================================\n\n";
  
  report += "SUMMARY METRICS\n";
  report += "-----------------\n";
  report += `Total Contracts Analyzed: ${data.summary.totalContracts}\n`;
  report += `Linked Contracts: ${data.summary.linkedContracts}\n`;
  report += `Shared Counterparties: ${data.summary.sharedCounterparties}\n`;
  report += `Average Risk Score: ${(data.summary.avgRiskScore * 100).toFixed(1)}%\n\n`;
  
  report += "DETAILED CONTRACT RELATIONSHIPS\n";
  report += "================================\n\n";
  
  data.contractPairs.forEach((pair, index) => {
    report += `[${index + 1}] ${pair.contractA} ↔ ${pair.contractB}\n`;
    report += `    Shared Counterparty: ${pair.sharedCounterparty}\n`;
    report += `    Risk Level: ${pair.riskLevel.toUpperCase()} (Score: ${(pair.riskScore * 100).toFixed(1)}%)\n`;
    report += `    Transactions: ${pair.transactionCount} | Failures: ${pair.failureRate}% | Value: $${pair.totalValue.toLocaleString()}\n\n`;
    
    report += `    WHY THEY ARE LINKED:\n`;
    pair.reasons.forEach(reason => {
      report += `      • ${reason}\n`;
    });
    
    report += `\n    RECOMMENDATIONS:\n`;
    pair.suggestions.forEach(suggestion => {
      report += `      → ${suggestion}\n`;
    });
    report += "\n" + "-".repeat(70) + "\n\n";
  });
  
  report += "\n===================================================\n";
  report += "End of Report | Generated by ContractNetAI\n";
  report += "===================================================\n";
  
  return report;
};
