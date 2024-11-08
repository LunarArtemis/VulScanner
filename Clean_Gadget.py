import re

# JavaScript keywords; immutable set
keywords = frozenset({'abstract', 'await', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                      'continue', 'debugger', 'default', 'delete', 'do', 'double', 'else', 'enum', 'export', 'extends',
                      'false', 'final', 'finally', 'float', 'for', 'function', 'goto', 'if', 'implements', 'import',
                      'in', 'instanceof', 'int', 'interface', 'let', 'long', 'native', 'new', 'null', 'package',
                      'private', 'protected', 'public', 'return', 'short', 'static', 'super', 'switch', 'synchronized',
                      'this', 'throw', 'throws', 'transient', 'true', 'try', 'typeof', 'var', 'void', 'volatile', 'while',
                      'with', 'yield'})

# holds known non-user-defined functions; immutable set
main_set = frozenset({'main'})
# arguments in main function; immutable set
main_args = frozenset({'argc', 'argv'})

def clean_gadget(gadget):
    fun_symbols = {}
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # regular expression to catch multi-line comment
    rx_comment = re.compile(r'\*/\s*$')
    # regular expression to find function name candidates
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    # regular expression to find variable name candidates
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()')

    cleaned_gadget = []

    for line in gadget:
        if rx_comment.search(line) is None:
            # remove all string literals (keep the quotes)
            nostrlit_line = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`', '""', line)
            # replace any non-ASCII characters with empty string
            ascii_line = re.sub(r'[^\x00-\x7f]', r'', nostrlit_line)

            # return, in order, all regex matches at string list; preserves order for semantics
            user_fun = rx_fun.findall(ascii_line)
            user_var = rx_var.findall(ascii_line)

            for fun_name in user_fun:
                if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                    if fun_name not in fun_symbols.keys():
                        fun_symbols[fun_name] = 'FUN' + str(fun_count)
                        fun_count += 1
                    ascii_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], ascii_line)

            for var_name in user_var:
                if len({var_name}.difference(keywords)) != 0 and len({var_name}.difference(main_args)) != 0:
                    if var_name not in var_symbols.keys():
                        var_symbols[var_name] = 'VAR' + str(var_count)
                        var_count += 1
                    ascii_line = re.sub(r'\b(' + var_name + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', var_symbols[var_name], ascii_line)

            cleaned_gadget.append(ascii_line)
    return cleaned_gadget

if __name__ == '__main__':
    test_gadget = ['function myFunc(var1, var2) {', 'var result = var1 + var2;', 'console.log(result);', '}']

    print(clean_gadget(test_gadget))