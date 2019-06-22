from operator import itemgetter

from src.intent.configs import IntentConfigs

config = IntentConfigs()
data = config.data.values
sorted(data, key=itemgetter(1))
idx = 0

intent_map = {}
for q, i in data:
    if i not in intent_map:
        idx = 0
    idx += 1
    intent_map[i] = idx

print("no. 의도 : 빈도수")
for k in zip(intent_map.keys(), intent_map.values()):
    print(config.intent_mapping[k[0]], '.', k[0], ' : ', k[1])
