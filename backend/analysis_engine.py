import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from gnn_inference import run_gnn_inference

logger = logging.getLogger(__name__)

class ContractAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.contract_names = {}
        self.transaction_data = None
        self.gnn_weights_path = Path(__file__).parent / "models" / "graphsage_model.pth"
        
    def analyze_csv(self, csv_content: str) -> Dict:
        """
        Main analysis function that processes CSV and returns structured results
        """
        try:
            # Parse CSV
            df = pd.read_csv(pd.io.common.StringIO(csv_content))
            logger.info(f"Loaded {len(df)} transactions")
            
            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Map columns to expected names
            column_mapping = {}
            for col in df.columns:
                if 'from' in col:
                    column_mapping[col] = 'from'
                elif 'to' in col:
                    column_mapping[col] = 'to'
                elif 'value' in col:
                    column_mapping[col] = 'value'
                elif 'error' in col:
                    column_mapping[col] = 'is_error'
            
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['from', 'to', 'value']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Clean data
            df['from'] = df['from'].astype(str).str.lower().str.strip()
            df['to'] = df['to'].astype(str).str.lower().str.strip()
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
            
            # Handle is_error column
            if 'is_error' not in df.columns:
                df['is_error'] = 0
            else:
                df['is_error'] = pd.to_numeric(df['is_error'], errors='coerce').fillna(0)
            
            # Remove invalid rows
            df = df[df['to'].notna() & (df['to'] != 'nan')]
            
            self.transaction_data = df
            
            # Generate readable contract names
            self._generate_contract_names(df)
            
            # Build graph
            self._build_graph(df)
            
            # Detect shared counterparties
            contract_pairs = self._detect_shared_counterparties(df)
            
            # Optional GNN inference for node risk
            gnn_scores: Optional[Dict[str, float]] = None
            try:
                gnn_scores = run_gnn_inference(csv_content, self.gnn_weights_path)
                logger.info("GNN inference succeeded; augmenting risk scores")
            except FileNotFoundError:
                logger.warning(f"GraphSAGE weights not found at {self.gnn_weights_path}; skipping GNN inference")
            except Exception as gnn_exc:
                logger.warning(f"GNN inference failed; falling back to heuristic scoring: {gnn_exc}")
                gnn_scores = None
            
            # Calculate risk scores
            analyzed_pairs = self._calculate_risk_scores(contract_pairs, df, gnn_scores)
            
            # Generate individual contract recommendations
            individual_contracts = self._generate_individual_recommendations(df, gnn_scores)
            
            # Generate summary
            summary = self._generate_summary(df, analyzed_pairs)
            
            return {
                'summary': summary,
                'contractPairs': analyzed_pairs,
                'individualContracts': individual_contracts
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _generate_contract_names(self, df: pd.DataFrame):
        """
        Generate human-readable contract names
        """
        prefixes = [
            "Supplier Agreement", "Vendor Contract", "Partnership Agreement",
            "Joint Venture", "Service Level Agreement", "Maintenance Contract",
            "Licensing Agreement", "Consultancy Contract", "Sales Agreement",
            "Distribution Contract", "Gateway Contract", "Payment Processing",
            "Infrastructure Service", "Technical Services", "Logistics Partner"
        ]
        
        suffixes = [chr(i) for i in range(65, 91)]  # A-Z
        
        all_addresses = pd.concat([df['from'], df['to']]).unique()
        
        name_index = 0
        for addr in all_addresses:
            if addr not in self.contract_names:
                prefix = prefixes[name_index % len(prefixes)]
                suffix = suffixes[(name_index // len(prefixes)) % len(suffixes)]
                self.contract_names[addr] = f"{prefix} {suffix}"
                name_index += 1
    
    def _build_graph(self, df: pd.DataFrame):
        """
        Build NetworkX graph from transaction data
        """
        for _, row in df.iterrows():
            fr = row['from']
            to = row['to']
            value = row['value']
            is_error = row['is_error']
            
            if self.graph.has_edge(fr, to):
                self.graph[fr][to]['weight'] += 1
                self.graph[fr][to]['value_sum'] += value
                self.graph[fr][to]['fail_count'] += (1 if is_error else 0)
            else:
                self.graph.add_edge(
                    fr, to,
                    weight=1,
                    value_sum=value,
                    fail_count=(1 if is_error else 0)
                )
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _detect_shared_counterparties(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect contracts that share counterparties and aggregate duplicates
        """
        user_contracts = defaultdict(set)

        for _, row in df.iterrows():
            user = row['from']
            contract = row['to']
            user_contracts[user].add(contract)

        pairs = {}
        for user, contracts in user_contracts.items():
            if len(contracts) < 2:
                continue
            for contract_a, contract_b in combinations(sorted(contracts), 2):
                key = (contract_a, contract_b)
                if key not in pairs:
                    pairs[key] = {
                        'contractA': contract_a,
                        'contractB': contract_b,
                        'sharedCounterparties': set(),
                        'contractA_name': self.contract_names.get(contract_a, contract_a[:10]),
                        'contractB_name': self.contract_names.get(contract_b, contract_b[:10]),
                    }
                pairs[key]['sharedCounterparties'].add(user)

        contract_pairs = []
        for idx, (_, data) in enumerate(pairs.items(), start=1):
            shared_list = sorted(data['sharedCounterparties'])
            contract_pairs.append({
                'id': idx,
                'contractA': data['contractA'],
                'contractB': data['contractB'],
                'sharedCounterparty': shared_list[0] if shared_list else None,
                'sharedCounterparties': shared_list,
                'contractA_name': data['contractA_name'],
                'contractB_name': data['contractB_name'],
                'counterparty_name': self.contract_names.get(shared_list[0], shared_list[0][:10]) if shared_list else "",
            })

        logger.info(f"Detected {len(contract_pairs)} unique contract pairs with shared counterparties")
        return contract_pairs
    
    def _calculate_risk_scores(self, contract_pairs: List[Dict], df: pd.DataFrame, gnn_scores: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Calculate risk scores for each contract pair
        """
        analyzed_pairs = []
        
        for pair in contract_pairs[:50]:  # Limit to top 50 pairs
            contract_a = pair['contractA']
            contract_b = pair['contractB']
            counterparties = pair.get('sharedCounterparties', [])
            shared_count = len(counterparties)
            shared_label = pair.get('counterparty_name') or (counterparties[0] if counterparties else "")
            
            # Get transaction stats for each contract
            stats_a = self._get_contract_stats(contract_a, df)
            stats_b = self._get_contract_stats(contract_b, df)
            
            # Calculate combined stats
            total_transactions = stats_a['count'] + stats_b['count']
            total_value = stats_a['total_value'] + stats_b['total_value']
            avg_failure_rate = (stats_a['failure_rate'] + stats_b['failure_rate']) / 2
            
            # Risk scoring (heuristic)
            risk_score = 0.0
            reasons = []
            
            # Factor 1: Shared counterparty dependency (0.3)
            if shared_count > 0:
                risk_score += 0.3
                if shared_count == 1:
                    reasons.append(f"Shared counterparty dependency on {shared_label}")
                else:
                    reasons.append(f"{shared_count} shared counterparties: {', '.join(counterparties[:3])}{'...' if shared_count > 3 else ''}")
            
            # Factor 2: Transaction failure rate (0.25)
            if avg_failure_rate > 10:
                risk_score += 0.25
                reasons.append(f"High combined failure rate {avg_failure_rate:.1f}% across {contract_a[:8]} / {contract_b[:8]}")
            elif avg_failure_rate > 5:
                risk_score += 0.15
                reasons.append(f"Moderate combined failure rate {avg_failure_rate:.1f}%")
            else:
                reasons.append(f"Low combined failure rate {avg_failure_rate:.1f}%")
            
            # Factor 3: Financial exposure (0.25)
            if total_value > 2000000:
                risk_score += 0.25
                reasons.append(f"High financial exposure ${total_value:,.0f} across pair")
            elif total_value > 1000000:
                risk_score += 0.15
                reasons.append(f"Moderate financial exposure ${total_value:,.0f}")
            else:
                reasons.append(f"Managed financial exposure ${total_value:,.0f}")
            
            # Factor 4: Circular dependencies (0.15)
            if self._has_circular_dependency(contract_a, contract_b):
                risk_score += 0.15
                reasons.append("Circular dependency pattern detected")
            
            # Factor 5: Transaction frequency (0.05)
            if total_transactions > 300:
                risk_score += 0.05
                reasons.append(f"High transaction volume ({total_transactions} tx)")
            
            # If GNN scores are available, blend in average of the two contracts
            if gnn_scores:
                gnn_vals = []
                if contract_a in gnn_scores:
                    gnn_vals.append(gnn_scores[contract_a])
                if contract_b in gnn_scores:
                    gnn_vals.append(gnn_scores[contract_b])
                if gnn_vals:
                    gnn_avg = np.mean(gnn_vals)
                    risk_score = max(risk_score, float(round(gnn_avg, 2)))
                    reasons.append(f"GNN risk signal: {gnn_avg:.2f}")

            # De-duplicate reasons for clarity
            reasons = list(dict.fromkeys(reasons))

            # Determine risk level
            if risk_score >= 0.7:
                risk_level = 'high'
            elif risk_score >= 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Generate suggestions with contract-specific context
            has_circular = self._has_circular_dependency(contract_a, contract_b)
            suggestions = self._generate_suggestions(
                risk_level, reasons, shared_label or "shared counterparties",
                pair['contractA_name'], pair['contractB_name'],
                total_value, avg_failure_rate, has_circular, total_transactions,
                stats_a, stats_b
            )
            
            analyzed_pairs.append({
                'id': pair['id'],
                'contractA': pair['contractA_name'],
                'contractB': pair['contractB_name'],
                'sharedCounterparty': shared_label,
                'sharedCounterparties': counterparties,
                'riskLevel': risk_level,
                'riskScore': round(risk_score, 2),
                'reasons': reasons,
                'suggestions': suggestions,
                'transactionCount': total_transactions,
                'failureRate': round(avg_failure_rate, 1),
                'totalValue': int(total_value)
            })
        
        # Sort by risk score (highest first)
        analyzed_pairs.sort(key=lambda x: x['riskScore'], reverse=True)
        
        return analyzed_pairs
    
    def _get_contract_stats(self, contract: str, df: pd.DataFrame) -> Dict:
        """
        Get statistics for a specific contract
        """
        contract_txs = df[df['to'] == contract]
        
        count = len(contract_txs)
        total_value = contract_txs['value'].sum()
        failures = contract_txs['is_error'].sum()
        failure_rate = (failures / count * 100) if count > 0 else 0
        
        return {
            'count': count,
            'total_value': total_value,
            'failure_rate': failure_rate
        }
    
    def _has_circular_dependency(self, contract_a: str, contract_b: str) -> bool:
        """
        Check if two contracts have circular dependency
        """
        try:
            # Check if there's a path from A to B and B to A
            has_path_ab = nx.has_path(self.graph, contract_a, contract_b)
            has_path_ba = nx.has_path(self.graph, contract_b, contract_a)
            return has_path_ab and has_path_ba
        except Exception:
            return False
    
    def _generate_individual_recommendations(self, df: pd.DataFrame, gnn_scores: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Generate individual contract-specific recommendations and risk profiles
        """
        individual_contracts = {}
        
        # Get all unique contracts
        all_contracts = pd.concat([df['from'], df['to']]).unique()
        
        for contract in all_contracts:
            # Get contract stats
            stats = self._get_contract_stats(contract, df)
            
            recommendations = []
            issues = []
            risk_score = 0.0
            
            # Issue 1: High failure rate
            if stats['failure_rate'] > 15:
                issues.append(f"Critical: {stats['failure_rate']:.1f}% transaction failure rate")
                risk_score += 0.35
                recommendations.append(f"URGENT: Debug {contract} - Transaction failure rate of {stats['failure_rate']:.1f}% is critical")
                recommendations.append(f"Run comprehensive diagnostics on {contract} transaction processing")
            elif stats['failure_rate'] > 10:
                issues.append(f"High: {stats['failure_rate']:.1f}% transaction failure rate")
                risk_score += 0.25
                recommendations.append(f"Investigate {contract} error patterns - {stats['failure_rate']:.1f}% failure rate detected")
            elif stats['failure_rate'] > 5:
                issues.append(f"Moderate: {stats['failure_rate']:.1f}% transaction failure rate")
                risk_score += 0.15
                recommendations.append(f"Monitor {contract} for emerging reliability issues")
            
            # Issue 2: Excessive transaction volume
            if stats['count'] > 500:
                issues.append(f"High volume: {stats['count']} transactions processed")
                risk_score += 0.2
                recommendations.append(f"Implement rate limiting and load balancing for {contract}")
                recommendations.append(f"Batch {contract} operations to reduce per-transaction overhead")
            elif stats['count'] > 200:
                issues.append(f"Moderate volume: {stats['count']} transactions processed")
                risk_score += 0.1
                recommendations.append(f"Consider transaction bundling for {contract} to optimize gas costs")
            
            # Issue 3: High financial exposure
            if stats['total_value'] > 5000000:
                issues.append(f"Critical exposure: ${stats['total_value']:,.0f} in transactions")
                risk_score += 0.2
                recommendations.append(f"Set strict spending limits on {contract} (${stats['total_value']:,.0f} at risk)")
                recommendations.append(f"Implement multi-signature approval for {contract} transactions >$100k")
            elif stats['total_value'] > 2000000:
                issues.append(f"High exposure: ${stats['total_value']:,.0f} in transactions")
                risk_score += 0.15
                recommendations.append(f"Enable real-time monitoring for {contract} (${stats['total_value']:,.0f} in transactions)")
            elif stats['total_value'] > 1000000:
                issues.append(f"Moderate exposure: ${stats['total_value']:,.0f} in transactions")
                risk_score += 0.1
                recommendations.append(f"Review monthly transaction limits for {contract}")
            
            # Issue 4: Counterparty concentration
            counterparties = self._get_contract_counterparties(contract, df)
            if len(counterparties) == 1:
                issues.append(f"Single counterparty dependency: {counterparties[0]}")
                risk_score += 0.15
                recommendations.append(f"Diversify {contract} - currently depends solely on {counterparties[0]}")
                recommendations.append(f"Establish backup relationships for {contract} operational continuity")
            elif len(counterparties) < 3:
                issues.append(f"Limited counterparty diversity: {len(counterparties)} partners")
                risk_score += 0.08
                recommendations.append(f"Expand {contract} counterparty base (currently {len(counterparties)} partners)")
            
            # Issue 5: Circular dependency involvement
            involved_in_circular = False
            for other_contract in all_contracts:
                if other_contract != contract and self._has_circular_dependency(contract, other_contract):
                    involved_in_circular = True
                    issues.append(f"Circular dependency with {other_contract}")
                    risk_score += 0.1
                    recommendations.append(f"Break circular dependency between {contract} and {other_contract}")
                    break
            
            # GNN-based risk signal
            if gnn_scores and contract in gnn_scores:
                gnn_score = gnn_scores[contract]
                risk_score = max(risk_score, float(round(gnn_score, 2)))
                if gnn_score > 0.7:
                    issues.append(f"GNN model flagged {contract} as high-risk")
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = 'high'
            elif risk_score >= 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Ensure uniqueness and limit to 5 recommendations
            recommendations = list(dict.fromkeys(recommendations))[:5]
            
            individual_contracts[contract] = {
                'contractName': contract,
                'riskScore': round(risk_score, 2),
                'riskLevel': risk_level,
                'failureRate': stats['failure_rate'],
                'totalValue': stats['total_value'],
                'transactionCount': stats['count'],
                'recommendations': recommendations,
                'issues': issues
            }
        
        return individual_contracts
    
    def _get_contract_counterparties(self, contract: str, df: pd.DataFrame) -> List[str]:
        """
        Get all unique counterparties (addresses) a contract interacts with
        """
        # Find all 'to' addresses when this contract is 'from'
        to_addresses = set(df[df['from'] == contract]['to'].unique())
        # Find all 'from' addresses when this contract is 'to'
        from_addresses = set(df[df['to'] == contract]['from'].unique())
        
        counterparties = list(to_addresses | from_addresses)
        return sorted(counterparties)
    
    def _generate_suggestions(self, risk_level: str, reasons: List[str], counterparty: str, 
                                contract_a_name: str, contract_b_name: str,
                                total_value: float, failure_rate: float, 
                                has_circular: bool, tx_count: int,
                                stats_a: Dict, stats_b: Dict) -> List[str]:
        """
        Generate actionable suggestions based on specific risk factors and contract characteristics
        """
        suggestions = []
        
        # Tailor suggestions based on individual contract performance
        if stats_a['failure_rate'] > 10 or stats_b['failure_rate'] > 10:
            worst_contract = contract_a_name if stats_a['failure_rate'] > stats_b['failure_rate'] else contract_b_name
            worst_rate = max(stats_a['failure_rate'], stats_b['failure_rate'])
            suggestions.append(f"Priority: Fix {worst_rate:.1f}% error rate in {worst_contract}")
            suggestions.append(f"Run diagnostics on {worst_contract} to identify transaction failure patterns")
        elif failure_rate > 5:
            suggestions.append(f"Monitor transaction reliability for {contract_a_name} and {contract_b_name}")
        
        # Value-based recommendations with specifics
        if total_value > 2000000:
            high_val_contract = contract_a_name if stats_a['total_value'] > stats_b['total_value'] else contract_b_name
            suggestions.append(f"Implement transaction limits on {high_val_contract} (${total_value:,.0f} at risk)")
            suggestions.append(f"Enable real-time alerts for {counterparty} transactions exceeding $50k")
        elif total_value > 1000000:
            suggestions.append(f"Set spending caps for {contract_a_name}↔{contract_b_name} via {counterparty}")
        elif total_value > 500000:
            suggestions.append(f"Review monthly spending trends across {counterparty} interactions")
        
        # Circular dependency handling
        if has_circular:
            suggestions.append(f"Break circular flow between {contract_a_name} and {contract_b_name} to prevent deadlocks")
            suggestions.append(f"Introduce intermediary contract to decouple {contract_a_name}/{contract_b_name} dependency")
        
        # Volume-based optimization
        if tx_count > 300:
            suggestions.append(f"Batch {tx_count} transactions between {contract_a_name} and {contract_b_name} to reduce gas costs")
        elif tx_count > 150:
            suggestions.append(f"Consider transaction bundling for {counterparty} operations")
        
        # Counterparty concentration risk
        if risk_level == 'high':
            suggestions.append(f"Critical: Reduce dependency on {counterparty} for {contract_a_name} and {contract_b_name}")
            suggestions.append(f"Establish backup providers to mitigate {counterparty} single point of failure")
        elif risk_level == 'medium':
            suggestions.append(f"Diversify counterparty base; {counterparty} handles too much volume")
            suggestions.append(f"Quarterly performance review of {counterparty} SLA compliance")
        else:
            suggestions.append(f"Continue quarterly monitoring of {counterparty} relationship health")
        
        # Contract-specific operational advice
        if stats_a['count'] > stats_b['count'] * 2:
            suggestions.append(f"Balance load: {contract_a_name} handles {stats_a['count']} tx vs {contract_b_name}'s {stats_b['count']}")
        
        # Ensure uniqueness and limit
        suggestions = list(dict.fromkeys(suggestions))
        return suggestions[:5]
    
    def _generate_summary(self, df: pd.DataFrame, analyzed_pairs: List[Dict]) -> Dict:
        """
        Generate summary statistics
        """
        unique_contracts = pd.concat([df['from'], df['to']]).nunique()
        linked_contracts = len(set([p['contractA'] for p in analyzed_pairs] + [p['contractB'] for p in analyzed_pairs]))
        shared_counterparties = len(set([p['sharedCounterparty'] for p in analyzed_pairs]))
        
        avg_risk_score = np.mean([p['riskScore'] for p in analyzed_pairs]) if analyzed_pairs else 0
        
        return {
            'totalContracts': unique_contracts,
            'linkedContracts': min(linked_contracts, unique_contracts),
            'sharedCounterparties': shared_counterparties,
            'avgRiskScore': round(avg_risk_score, 2)
        }
    
    def generate_csv_report(self, results: Dict) -> str:
        """
        Generate CSV report from analysis results
        """
        header = "Contract A,Contract B,Shared Counterparty,Risk Level,Risk Score,Transaction Count,Failure Rate (%),Total Value ($)\n"
        rows = []
        
        for pair in results['contractPairs']:
            rows.append(
                f'"{pair["contractA"]}",'
                f'"{pair["contractB"]}",'
                f'"{pair["sharedCounterparty"]}",'
                f'{pair["riskLevel"]},'
                f'{pair["riskScore"]},'
                f'{pair["transactionCount"]},'
                f'{pair["failureRate"]},'
                f'{pair["totalValue"]}'
            )
        
        return header + "\n".join(rows)
    
    def generate_txt_report(self, results: Dict) -> str:
        """
        Generate TXT report from analysis results
        """
        report = "="*70 + "\n"
        report += "         CONTRACTNETAI ANALYSIS REPORT\n"
        report += "         Cross-Contract Intelligence Report\n"
        report += "="*70 + "\n\n"
        
        summary = results['summary']
        report += "SUMMARY METRICS\n"
        report += "-" * 50 + "\n"
        report += f"Total Contracts Analyzed: {summary['totalContracts']}\n"
        report += f"Linked Contracts: {summary['linkedContracts']}\n"
        report += f"Shared Counterparties: {summary['sharedCounterparties']}\n"
        report += f"Average Risk Score: {summary['avgRiskScore']*100:.1f}%\n\n"
        
        report += "DETAILED CONTRACT RELATIONSHIPS\n"
        report += "="*70 + "\n\n"
        
        for idx, pair in enumerate(results['contractPairs'], 1):
            report += f"[{idx}] {pair['contractA']} ↔ {pair['contractB']}\n"
            report += f"    Shared Counterparty: {pair['sharedCounterparty']}\n"
            report += f"    Risk Level: {pair['riskLevel'].upper()} (Score: {pair['riskScore']*100:.0f}%)\n"
            report += f"    Transactions: {pair['transactionCount']} | Failures: {pair['failureRate']}% | Value: ${pair['totalValue']:,}\n\n"
            
            report += "    WHY THEY ARE LINKED:\n"
            for reason in pair['reasons']:
                report += f"      • {reason}\n"
            
            report += "\n    RECOMMENDATIONS:\n"
            for suggestion in pair['suggestions']:
                report += f"      → {suggestion}\n"
            
            report += "\n" + "-"*70 + "\n\n"
        
        report += "\n" + "="*70 + "\n"
        report += "End of Report | Generated by ContractNetAI\n"
        report += "="*70 + "\n"
        
        return report
