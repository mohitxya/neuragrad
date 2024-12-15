import math
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self.grad=0.0 #Doesn't change the output. Has no effect.
        self._prev=set(_children)
        self._backward=lambda: None
        self._op=_op
        self.label=label
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self,other):
        other=other if isinstance(other, Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward #we store the function here.
        return out
        
    def __sub__(self, other):
        return self+(-other)
        
    def __mul__(self,other):
        other=other if isinstance(other, Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out
        
    def __rmul__(self, other):
        return self*other
        
    def __pow__(self, other):
        assert isinstance(other, (int,float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,),f'**{other}')  
        def _backward():
            self.grad+=other*(self.data**(other-1))*out.grad #first part is the local derivative only. total is derivative wrt final o
        out._backward=_backward
        return out
        
    def __truediv__(self,other):
        return self*other**-1

    def __radd__(self,other):
        return self+other
        
    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=Value(t,(self,),'tanh')
        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        return out
    def exp(self):
        x=self.data
        out=Value(math.exp(x), (self,),'exp')
        def _backward():
            self.grad+=out.data*out.grad 
        out._backward=_backward
        return out
    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        topo
        self.grad=1.0
        for node in reversed(topo):
            node._backward()
        