import os
import json
import esprima
from tqdm import tqdm
import pandas as pd

def extract_code_gadgets(code):
    gadgets = []
    
    try:
        ast = esprima.parseScript(code, {
            'range': True,
            'tolerant': True,
            'jsx': False
        })
    except Exception as e:
        print(f"🔥 Parsing failed: {str(e)}")
        return gadgets

    sensitive_apis = [
        "fs.readFile", "fs.readFileSync", "fs.createReadStream",
        "fs.writeFile", "fs.writeFileSync", "fs.appendFile", "fs.appendFileSync",
        "fs.unlink", "fs.unlinkSync", "fs.rm", "fs.rmSync",
        "fs.readdir", "fs.readdirSync",
        "fs.existsSync", "fs.access", "fs.accessSync",
        "path.join", "path.resolve", "path.normalize",
        "child_process.exec", "child_process.execSync",
        "child_process.spawn", "child_process.spawnSync",
        "child_process.fork",
        "eval", "Function", "require", "import",
        "moment.locale"
    ]


    class GadgetExtractor(esprima.NodeVisitor):
        def __init__(self):
            self.sensitive_operations = []
            self.user_inputs = set()
            self.debug_log = []

        def get_source_code(self, node):
            """Safely extract source code with range validation"""
            if not hasattr(node, 'range') or node.range is None:
                return None
            start, end = node.range
            if start is None or end is None:
                return None
            return code[start:end]

        def visit_VariableDeclarator(self, node):
            try:
                if node.init:
                    source = self.get_source_code(node.init)
                    if source and any(s in source for s in ['req.query', 'req.params', 'req.body']):
                        var_name = node.id.name
                        self.user_inputs.add(var_name)
                        self.sensitive_operations.append({
                            'node': node,
                            'type': 'user_input',
                            'source': self.get_source_code(node)
                        })
            except Exception as e:
                print(f"VariableDeclarator error: {str(e)}")
            self.generic_visit(node)

        def visit_CallExpression(self, node):
            try:
                # Detect require() with user input
                if hasattr(node.callee, 'name') and node.callee.name == 'require':
                    arg_source = self.get_source_code(node.arguments[0])
                    if arg_source and any(var in arg_source for var in self.user_inputs):
                        self.sensitive_operations.append({
                            'node': node,
                            'type': 'dynamic_require',
                            'source': self.get_source_code(node)
                        })

                # Detect sensitive API calls
                if node.callee.type == 'MemberExpression':
                    obj = self.unwind_member_expression(node.callee)
                    method_call = f"{obj}.{node.callee.property.name}"
                    
                    if method_call in sensitive_apis:
                        self.sensitive_operations.append({
                            'node': node,
                            'type': method_call,
                            'source': self.get_source_code(node)
                        })
            except Exception as e:
                print(f"CallExpression error: {str(e)}")
            self.generic_visit(node)

        def unwind_member_expression(self, node):
            """Robust member expression unwinding"""
            parts = []
            try:
                while node.type == 'MemberExpression':
                    parts.append(node.property.name)
                    node = node.object
                base = node.name if hasattr(node, 'name') else self.get_source_code(node)
                return f"{base}.{'.'.join(reversed(parts))}" if parts else base
            except Exception:
                return "UnknownExpression"

    extractor = GadgetExtractor()
    extractor.visit(ast)

    # Safely create gadgets
    for op in extractor.sensitive_operations:
        # Validate node and range
        if not op['source']:
            continue
            
        node = op['node']
        if not hasattr(node, 'range') or node.range is None:
            continue
            
        start, end = node.range
        if start is None or end is None:
            continue

        # Calculate line numbers safely
        try:
            start_line = code[:start].count('\n') + 1
            end_line = code[:end].count('\n') + 1
            context_start = max(0, start_line - 3)  # 0-based index
            context_end = end_line + 1  # +2 lines after
            lines = code.split('\n')
            gadget = {
                'type': op['type'],
                'source': op['source'],
                'context': lines[context_start:context_end],
                'line_numbers': (start_line, end_line)
            }
            gadgets.append(gadget)
        except Exception as e:
            print(f"Skipping gadget due to error: {str(e)}")

    return gadgets


# main function
if __name__ == '__main__':
    # Configuration
    INPUT_CSV = "../Datasets/output_cleaned.csv"  # Path to your CSV file
    OUTPUT_FILE = "processed_gadgets_3.csv"  # Output file for results

    def process_csv_dataset():
        # Load the CSV dataset
        try:
            df = pd.read_csv(INPUT_CSV)
        except Exception as e:
            print(f"❌ Error loading CSV file: {str(e)}")
            return

        # Process each row in the dataset
        results = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
            try:
                code = row['code']
                label = row['label']
                
                # Extract gadgets
                gadgets = extract_code_gadgets(code)
                
                # Flatten gadgets into individual rows
                for gadget in gadgets:
                    # Join the context lines into a single string
                    gadget_context = "\n".join(gadget['context'])
                    results.append({
                        'gadget': gadget_context,
                        'label': label
                    })
            except Exception as e:
                print(f"❌ Error processing row {index}: {str(e)}")

        # Save results to a new CSV file
        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(OUTPUT_FILE, index=False)
            print(f"\n✅ Processing complete! Results saved to {OUTPUT_FILE}")
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")

    # Run the dataset processing
    process_csv_dataset()