import sys
import pandas as pd

if len(sys.argv) != 4:
    print('compare.py <results> <baseline> <output-to>')

print(f"""
    Generating comparisons on files

        benchmark-results: {sys.argv[1]}
        baseline: {sys.argv[2]}
    
    Outputting to: {sys.argv[3]}
""")

results = pd.read_csv(sys.argv[1])
baseline = pd.read_csv(sys.argv[2])

# TODO validate they are comparable

# Just to show it works
results['acc'] = results['acc'].apply(lambda x: x + 100)

comparisons = pd.DataFrame({
    'task': list(baseline['task']),
    'fold': list(baseline['fold']),
    'seed': list(baseline['seed']),
    'acc_diff': list(
        x - y for x, y in zip(results['acc'], baseline['acc'])
    ),
})

comparisons.to_csv(path=sys.argv[3])

