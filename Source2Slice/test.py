import esprima

def test():
    # testing with file input
    with open('SourceCode/vuln2.js') as f:
        test = f.read()
    parsed = esprima.parseScript(test)

    print(parsed)
    
    # testing the esprima tokenizer
    test = 'var x = 1;'
    tokens = esprima.tokenize(test)
    
    # for token in tokens:
    #     print(token)
        
if __name__ == '__main__':
    test()