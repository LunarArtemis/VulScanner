import re

# JavaScript/TypeScript keywords set
keywords = frozenset({'abstract', 'await', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                      'continue', 'debugger', 'default', 'delete', 'do', 'double', 'else', 'enum', 'export', 'extends',
                      'false', 'final', 'finally', 'float', 'for', 'function', 'goto', 'if', 'implements', 'import',
                      'in', 'instanceof', 'int', 'interface', 'let', 'long', 'native', 'new', 'null', 'package',
                      'private', 'protected', 'public', 'return', 'short', 'static', 'super', 'switch', 'synchronized',
                      'this', 'throw', 'throws', 'transient', 'true', 'try', 'typeof', 'var', 'void', 'volatile', 'while',
                      'with', 'yield'})

# Holds known common function names; immutable set (in place of 'main' in C++)
main_set = frozenset({'init', 'start'})

# JavaScript/TypeScript doesn’t have argc/argv, but you may add similar terms as needed.
main_args = frozenset({})

# Function to clean JavaScript/TypeScript gadget
def clean_gadget(gadget):
    # Dictionaries to map function/variable names to symbol names
    fun_symbols = {}
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # Regex to detect multi-line comment end
    rx_comment = re.compile(r'\*/\s*$')
    # Regex to find function names in JavaScript (e.g., `function name(...)`, `const name = (...) => {}`)
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    # Regex for variable names in JavaScript (not followed by a function definition)
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')

    # Final cleaned gadget output to return
    cleaned_gadget = []

    for line in gadget:
        # Process line if it’s not a header or multi-line comment
        if rx_comment.search(line) is None:
            # Remove all string literals (keep the quotes)
            nostrlit_line = re.sub(r'".*?"', '""', line)
            # Remove character literals (JavaScript doesn't have these)
            nocharlit_line = re.sub(r"'.*?'", "''", nostrlit_line)
            # Remove any non-ASCII characters
            ascii_line = re.sub(r'[^\x00-\x7f]', r'', nocharlit_line)

            # Find function and variable names in line
            user_fun = rx_fun.findall(ascii_line)
            user_var = rx_var.findall(ascii_line)

            # Map function names to unique symbols
            for fun_name in user_fun:
                if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                    if fun_name not in fun_symbols:
                        fun_symbols[fun_name] = 'FUN' + str(fun_count)
                        fun_count += 1
                    # Replace function names (only when followed by `(`)
                    ascii_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], ascii_line)

            # Map variable names to unique symbols
            for var_name in user_var:
                if len({var_name}.difference(keywords)) != 0 and len({var_name}.difference(main_args)) != 0:
                    if var_name not in var_symbols:
                        var_symbols[var_name] = 'VAR' + str(var_count)
                        var_count += 1
                    # Replace variable names (not followed by `(`)
                    ascii_line = re.sub(r'\b(' + var_name + r')\b(?!\s*\()', var_symbols[var_name], ascii_line)

            cleaned_gadget.append(ascii_line)

    return cleaned_gadget

if __name__ == '__main__':
    # Example JavaScript gadget to clean
    test_gadget = [
        'const fs = require("fs");',
        'function processFile(file) {',
        '  let data = fs.readFileSync(file, "utf8");',
        '  console.log(data);',
        '}'
    ]
    
    test_gadgetline = ['function(File file, Buffer buff)', 'this is a comment test */']

    print(clean_gadget(test_gadget))
    print(clean_gadget(test_gadgetline))