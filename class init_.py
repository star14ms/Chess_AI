class A:

    def __init__(self, a, b):
        self.x = a
        self.y = b
        print(type(self) == B)

class B(A):
    pass

class C(A):
    pass


b = B(1, 2)
c = C(3, 4)
print(b.x, b.y)
print(c.x, c.y)

print(1 == 1 or 2 == 2)
print(1 == 1 & 2 == 2)
print((1 == 1) & (2 == 2))