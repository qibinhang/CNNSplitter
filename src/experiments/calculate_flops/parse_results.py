with open(f'./flops.log', 'r') as f:
    results = []
    for each_line in f.readlines():
        if each_line.startswith('Module: '):
            module_flops = each_line.strip()[len('Module: '):-1]
            print(module_flops)
