import io
import contextlib
import re
import torch
from scrapper import parse_page
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
torch.cuda.empty_cache()

model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

def format_prompt(problem):
    return f"""SYSTEM: You are a Python coding assistant. 
    Write ONLY the Python function to solve the problem. 
    No explanations, comments, or text besides code.

### Problem:
{problem}

### Solution Code:
def solve():
"""


def generate_solution(prompt, num_samples=5, max_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        max_new_tokens=max_tokens,
        num_return_sequences=num_samples
    )
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

def clean_code(code):
    # Remove all comments
    code = re.sub(r'#.*', '', code)
    
    # Remove docstrings
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    # Standard replacements
    replacements = {
        '\u00a0': ' ',   # Non-breaking space
        '≤': '<=', '≥': '>=', '≠': '!=',
        '÷': '/', '×': '*', '−': '-', '∗': '*'
    }
    for k, v in replacements.items():
        code = code.replace(k, v)
        
    return code.strip()


def run_test(user_code, inputs, outputs):
    input_data = '\n'.join(inputs)
    expected_output = '\n'.join(outputs)

    full_code = f"""
import io
import contextlib
output = []
input_data = '''{input_data}'''
input_lines = input_data.strip().split('\\n')
input = lambda: input_lines.pop(0)
with contextlib.redirect_stdout(io.StringIO()) as f:
    solve()
    output = f.getvalue().strip().split('\\n')
"""

    full_code += """
assert output == expected_output, f"Expected: {expected_output}, Got: {output}"
"""

    try:
        # Clean up any invisible unicode whitespace in the generated code
        user_code_clean = clean_code(user_code)
        print("Generated code:\n", user_code_clean)
        local_env = {
            "contextlib": contextlib,
            "io": io,
            "expected_output": expected_output,
        }
        exec(user_code_clean + "\n" + full_code, local_env)
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def extract_code(text):
    """
    Extracts code between `````` blocks or the first function.
    """
    # Try to find code fences first
    code_blocks = re.findall(r'``````', text, re.DOTALL)  # Added 'text' and flags
    if code_blocks:
        return code_blocks[0].strip()  # Take first match and strip whitespace
    
    # Fallback: find first function definition (fixed regex)
    func_match = re.search(r'(def solve\(\):[\s\S]*)', text)  # Fixed "$$$$" → "\(\)"
    if func_match:
        return func_match.group(1).strip()
    
    return text

def main(url):
    problem, inputs, outputs = parse_page(url)
    prompt = format_prompt(problem)
    samples = generate_solution(prompt)
    for i, solution in enumerate(samples):
        # Extract clean code
        solution_code = extract_code(solution)
        
        # Basic validation
        if not solution_code.strip().startswith('def solve():'):
            print(f"⚠️ Sample {i+1}: No valid function found in:")
            print(solution)
            continue
            
        # Run tests
        print(f"Sample {i+1} Result:")
        test_case = problem["public_tests"]
        passed = run_test(solution_code, inputs, outputs)
        print("✅ Passed" if passed else "❌ Failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter url to get response")
    parser.add_argument("--url", type=str, required=True, help="Enter problem link from codeforces")
    args = parser.parse_args()
    main(args.url)

