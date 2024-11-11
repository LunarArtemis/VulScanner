import re

# JavaScript keywords; immutable set, REMOVED 'function' and 'const' as they are not considered keywords
'''
=========================== REMOVED 'function' and 'const' ===========================
'''
keywords = frozenset({'abstract', 'await', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                      'continue', 'debugger', 'default', 'delete', 'do', 'double', 'else', 'enum', 'export', 'extends',
                      'false', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 'import',
                      'in', 'instanceof', 'int', 'interface', 'let', 'long', 'native', 'new', 'null', 'package',
                      'private', 'protected', 'public', 'return', 'short', 'static', 'super', 'switch', 'synchronized',
                      'this', 'throw', 'throws', 'transient', 'true', 'try', 'typeof', 'var', 'void', 'volatile', 'while',
                      'with', 'yield'})


# Holds known common function names; immutable set (in place of 'main' in C++)
main_set = frozenset({'init', 'start'})

# JavaScript/TypeScript doesn’t have argc/argv, but you may add similar terms as needed.
main_args = frozenset({})

# Compile regex patterns
rx_comment = re.compile(r'\*/\s*$')
rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()')
rx_str_lit = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`')
rx_non_ascii = re.compile(r'[^\x00-\x7f]')

def clean_string_literals_and_non_ascii(line):
    # Remove all string literals
    line = rx_str_lit.sub('""', line)
    # Replace any non-ASCII characters with empty string
    return rx_non_ascii.sub('', line)

def clean_gadget(gadget):
    fun_symbols = {}
    var_symbols = {}

    fun_count = 1
    var_count = 1

    cleaned_gadget = []

    for line in gadget:
        if not rx_comment.search(line):
            line = clean_string_literals_and_non_ascii(line)
            user_fun = rx_fun.findall(line)
            user_var = rx_var.findall(line)

            for fun_name in user_fun:
                if fun_name not in main_set | keywords:
                    if fun_name not in fun_symbols:
                        fun_symbols[fun_name] = f'FUN{fun_count}'
                        fun_count += 1
                    line = re.sub(rf'\b{fun_name}\b(?=\s*\()', fun_symbols[fun_name], line)

            for var_name in user_var:
                if var_name not in keywords | main_args:
                    if var_name not in var_symbols:
                        var_symbols[var_name] = f'VAR{var_count}'
                        var_count += 1
                    line = re.sub(rf'\b{var_name}\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', var_symbols[var_name], line)

            cleaned_gadget.append(line)
    return cleaned_gadget

if __name__ == '__main__':
    test_gadget = [
        'function myFunc(var1, var2) {',
        'var result = var1 + var2;',
        'console.log(result);',
        '}'
    ]
    test_gadget2 = [
        'function myFunc(var1, var2) {',
        'var result = var1 + var2;',
        'console.log(result);',
        '}',
        'function main() {',
        'var x = 5;',
        'var y = 10;',
        'myFunc(x, y);',
        '}'
    ]
    test_gadget3 = [
        "let s = {",
        "    a: null,",
        "    b: null,",
        "    uninit: null",
        "};",
        "",
        "s.a = 20;",
        "s.b = 20;",
        "s.uninit = 20;"
    ]
    test_gadgetline = ['function(File file, Buffer buff)', 'this is a comment test */']

    print(clean_gadget(test_gadget))
    print(clean_gadget(test_gadget2))
    print(clean_gadget(test_gadget3))
    print(clean_gadget(test_gadgetline))