import torch
ck = torch.load('/icislab/volume1/wyy/O1O-main/exps/o1o_mowod_task2_train/checkpoint0039.pth', map_location='cpu')
a = ck.get('args', None)
print('has args:', a is not None)
if a is not None:
    for k in ['exemplar_replay_selection', 'exemplar_replay_max_length',
              'dataset', 'PREV_INTRODUCED_CLS', 'CUR_INTRODUCED_CLS',
              'num_superclass', 'resume']:
        print('  ', k, '=', getattr(a, k, '<none>'))
    for k in dir(a):
        if 'replay' in k.lower():
            print('  replay-arg:', k, '=', getattr(a, k))
