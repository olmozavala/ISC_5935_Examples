from PyTest.mymodule import mysum

def test_mysum():
    assert mysum(3,4) == 7
    assert mysum(-1, 4) == 3
    assert mysum(-1, -3) == -4
