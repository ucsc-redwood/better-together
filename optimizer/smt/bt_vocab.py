"""AUTO-GENERATED from vocab.json by scripts/embed_vocab.py -- DO NOT EDIT.
Regenerate after changing vocab.json.
"""

# Solver CPU tiers, in cost-matrix column order (the orphaned 'super' tier is
# intentionally NOT here -- it is absent from the z3 tier list today).
CPU_TIERS = ('little', 'medium', 'big')

# Solver core-type columns: the CPU tiers (display-cased) + the GPU column.
CORE_TYPES = ['Little', 'Medium', 'Big', 'GPU']

# Application stage counts.
APP_STAGES = {'tree': 7, 'cifar-dense': 9, 'cifar-sparse': 9}
