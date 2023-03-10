class myclass():
    def __init__(self, size):
        self.size = size

size_value = 5
myinstance = myclass(size_value)
def myfunction(myobj = myclass(1)):
    print(f"Specific attribure: {myobj.size}")
##

myfunction(myinstance)
