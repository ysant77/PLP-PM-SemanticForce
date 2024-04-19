import re

def extract_model_responses(text: str) -> list:
    # Pattern to match everything after "model" till it potentially hits another "user" or end of string
    pattern = r"model(.*?)(?=user|$)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    # Clean and return the matches
    return [match.strip() for match in matches if match.strip()]