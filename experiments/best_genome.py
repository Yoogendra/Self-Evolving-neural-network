BEST_GENOME = {
  'blocks': [
    {'out': 96, 'k': 3, 'act': 'leaky_relu', 'bn': True,  'drop': 0.0, 'pool': 'avg'},
    {'out': 32, 'k': 5, 'act': 'relu',      'bn': True,  'drop': 0.0, 'pool': 'max'},
    {'out': 48, 'k': 5, 'act': 'leaky_relu','bn': False, 'drop': 0.1, 'pool': 'max'},
    {'out': 128,'k': 3, 'act': 'relu',      'bn': True,  'drop': 0.1, 'pool': 'none'}
  ],
  'head': {'dense': 128, 'drop': 0.2},
  'opt': {'lr': 0.0015182875143617814, 'weight_decay': 0.0005}
}
