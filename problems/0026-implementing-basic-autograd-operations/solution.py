
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out=Value(self.data + other.data, (self, other), "+")
        
        def _backward():
          self.grad+=out.grad
          other.grad+=out.grad
        out._backward=_backward
        return out
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out=Value(self.data * other.data, (self, other), "*")
        def _backward():
          self.grad+=other.data * out.grad
          other.grad+=self.data * out.grad
        out._backward=_backward
        return out

    def __rmul__(self, other):
        # Redirects something like '2 * Value(3)' to 'Value(3) * 2'
        return self.__mul__(other)

    def relu(self):
        # Implement ReLU here
        out_data = self.data if self.data >= 0 else 0
        out=Value(out_data, (self,), "ReLU")
        def _backward():
          self.grad+=(out.data>0 )*out.grad
        out._backward=_backward
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()