
## Simplest exception
def f(x):
    return 1/x

try:
    # ... previous code
    f(x) # Call to a function that may raise an exception
    # ... more code
except Exception as e:
    print(e)
finally:
    print("This will always be executed!")

##
x = 0
if x == 0:
    raise Exception("Can't divide by zero!")
##

