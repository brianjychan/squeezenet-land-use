'''
Miscellaneous code that was utilized to move files around
'''

import os

# for dr in os.listdir('./data/train'):
#     if dr.startswith('.'):
#             continue
#     for filename in os.listdir('./data/train' + '/' + dr):
#         if filename.startswith('.'):
#             continue
#         if int(filename[-7:-4]) > 630:
#             print(filename)
#             print('./data/train' + '/' + dr + '/' + filename)
#             os.rename('./data/train' + '/' + dr + '/' + filename, './data/test' + '/' + dr + '/' + filename)

for dr in os.listdir('./data/train'):
    if dr.startswith('.'):
            continue
    if not os.path.exists('./data/validation/' + dr):
        print('./data/validation/' + dr)
        os.mkdir('./data/validation/' + dr)
    for filename in os.listdir('./data/train' + '/' + dr):
        if filename.startswith('.'):
            continue
        if int(filename[-7:-4]) <= 630 and int(filename[-7:-4]) > 560 :
            print(filename)
            print('./data/train' + '/' + dr + '/' + filename)
            os.rename('./data/train' + '/' + dr + '/' + filename, './data/validation' + '/' + dr + '/' + filename)
