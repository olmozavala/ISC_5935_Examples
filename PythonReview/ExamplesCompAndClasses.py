
## ------ Basic syntact
# print([f'My number is {x*2+1}' for x in range(10)])
# print((f'My number is {x*2+1}' for x in range(10)))
# x = {'a':1, 'b':2}
# print({f'a{x}':x for x in range(10)})
print([y*2 for y in [x for x in range(5)]])

## ------
print([f'My number is {x*2+1}' for x in range(10) if x > 4])

##
print(['a' if x > 4 else 'b' for x in range(10) ])

##

