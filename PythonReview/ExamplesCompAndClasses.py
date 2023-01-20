## ------ Comprehensions ---
# Basic syntacs 
print([f'My number is {x*2+1}' for x in range(10)])
# Nested  Comp
print([y*2 for y in [x for x in range(5)]])

## Filtering comp
print([f'My number is {x*2+1}' for x in range(10) if x > 4])

## Conditional selection 
print(['a' if x > 4 else 'b' for x in range(10) ])

## -------- Exceptions 

