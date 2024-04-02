
import math
import matplotlib.pyplot as plt
from graphviz import Digraph

class value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0
    self._prev = set(_children)
    self._op = _op
    self.label = label
    self._backward = lambda: None
    self._len_expression_graph = 1
  def __repr__(self):
    return f"Value (data = {self.data})"
  
  def __add__(self, other):
    other = other if isinstance(other, value) else value(other)
    out = value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out
  
  def __radd__(self, other):
    return self.__add__(other)
  
  def __neg__(self):
    return self * -1
  
  def __sub__(self, other):
    return self + (-other)
  
  def __rsub__(self, other):
    return value(other) - self


  def __mul__(self, other):
    other = other if isinstance(other, value) else value(other)
    out = value(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self.__mul__(other)  

  def __truediv__(self, other): # self / other
    return self * other ** -1 # (self) * (other^-1) precendence is set automatically
  
  def __rtruediv__(self, other): # other / self
    return value(other) * self ** -1

  def __pow__(self,other):
    assert isinstance(other, (int, float)), "only supporting int/float for now"
    out = value(self.data ** other, (self,), f'**{other}')
    def _backward():
      self.grad += other * self.data ** (other - 1) * out.grad
      # other.grad += self.data ** other * math.log(self.data) * out.grad
    out._backward = _backward
    return out

  def exp(self):
      x = self.data
      out = value(math.exp(x), (self,), 'exp')
      def _backward():
        self.grad += out.data * out.grad
      out._backward = _backward
      return out

  def tanh(self):
    out = value(math.tanh(self.data), (self,), 'tanh')
    def _backward(): 
      self.grad += (1 - out.data**2) * out.grad
    out._backward = _backward
    return out
  
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = 1
    for node in reversed(topo):
      node._backward()
    self._len_expression_graph = len(topo)

  def reset_grad(self):
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        v.grad = 0.0
    build_topo(self)


def trace(root):
  # builds a set of all nodes and edges in the graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir':'LR'}) # Left to right graph

  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    dot.node(name=uid, label= "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record') 

    if n._op:
      dot.node(name=uid + n._op, label= n._op)
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot
