import re

class ClinicalEvaluator:
    """
    Implements RECIST 1.1 (Response Evaluation Criteria in Solid Tumors) 
    and other clinical heuristic checks.
    """
    
    @staticmethod
    def calculate_percent_change(current: float, previous: float) -> float:
        if previous == 0: return 0.0
        return ((current - previous) / previous) * 100

    @staticmethod
    def evaluate_response(current_size: float, baseline_size: float) -> str:
        """
        RECIST 1.1 Categories:
        - Progressive Disease (PD): >20% increase
        - Partial Response (PR): >30% decrease
        - Stable Disease (SD): Neither
        """
        change = ClinicalEvaluator.calculate_percent_change(current_size, baseline_size)
        
        if change >= 20:
            return "Progressive Disease (PD) - Intervention Required"
        elif change <= -30:
            return "Partial Response (PR) - Continue Treatment"
        else:
            return "Stable Disease (SD)"

    @staticmethod
    def extract_numeric_value(text: str) -> float:
        """Helper to pull numbers from messy clinical strings."""
        match = re.search(r"(\d+(\.\d+)?)", text)
        return float(match.group(1)) if match else 0.0