import os

icons = ['hammer', 'lightning', 'fire', 'ghost', 'key', 'cat', 'icecube', 'snowflake', 
          'turtle', 'milkbottle', 'eye', 'clown', 'cactus', 'gkey', 'cobweb', 'lightbulb', 
          'carrot', 'hand', 'bird', 'stopsign', 'igloo', 'lips', 'flower', 'exclamationmark',
          'car', 'lock', 'anchor', 'moon', 'man', 'clock', 'tree', 'heart', 'spider', 'stains', 
          'dolphin', 'apple', 'ladybug', 'trex', 'sun', 'cheese', 'questionmark', 'dog', 'horse', 
          'flyingdino', 'zebra', 'yinyang', 'sunglasses', 'skull', 'candle', 'snowman', 'leaf', 
          'drop', 'bomb', 'scissors', 'pencil', 'bullseye', 'clover']


for name in icons:
    os.mkdir(os.path.join('icons/', str(name)))