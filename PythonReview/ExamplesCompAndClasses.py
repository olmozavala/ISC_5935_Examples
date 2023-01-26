## ------ Comprehensions ---
# Basic syntacs 
print([f'My number is {x*2+1}' for x in range(10)])
# Nested  Comp
print([y*2 for y in [x for x in range(5)]])

## Filtering comp
print([f'My number is {x*2+1}' for x in range(10) if x > 4])

## Conditional selection 
print(['a' if x > 4 else 'b' for x in range(10) ])

## -------- Class
class Patient:
    name = "No name"
    age = -1

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def make_report(self):
        print(f"My name is {self.name} age: {self.age}")

p1 = Patient("Oscar", 67)
p2 = Patient("P2", 35)

p1.make_report()
p2.make_report()
print("Done!")

##Exceptions
def f(x):
    return 1/x

try:
    print("Before exp")
    f(1)
    print("After exp")
except Exception as e:
    print(f"The exception that happen is e:{e}")
finally:
    print("This code will always happen")

##
def f(x):
    if x == 0:
        raise Exception("Please do not send me 0!")
    return 1/x
##
f(0)

