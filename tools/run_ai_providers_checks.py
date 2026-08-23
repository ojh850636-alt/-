import importlib, json
m = importlib.import_module('ai_providers')
cases = [
    ({}, 'case_stub'),
    ({'mock': True, 'prompt': 'hi'}, 'case_mock'),
]
res = {}
for payload, name in cases:
    try:
        r = m.call_best_provider(payload)
        res[name] = r
    except Exception as e:
        res[name] = {'exception': str(e)}
with open('tools/ai_providers_test_results.txt','w',encoding='utf-8') as f:
    f.write(json.dumps(res, ensure_ascii=False, indent=2))
print('wrote tools/ai_providers_test_results.txt')
