# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 22:53:42 2021

@author: Andrea Bassi
"""
import numpy as np

def extend_to_Vector(method):
    """Class decorator to extend base method to the Vector objects"""  
    def wrapper(*args):
        first = args[0]
        second =  args[1]
        if isinstance(second, Vector):
            function = getattr(first.v, method.__name__)
            result = function(second.v)
        else:
            function = getattr(first.v, method.__name__)
            result = function(second)
        return Vector.from_array(result)
    return wrapper

class Vector():
    ''' Class for handling 3D vectors in which
        each element of the vector is a 2D numpy.array.
        It is an overstructure of numpy but keeps higher level code more readable.   
    '''
    
    def __init__(self,x,y,z):
        ''' x, y and z are 2D numpy.array'''
        self.v = np.array([z,y,x])
        
        
    def __repr__(self):
        return f'{self.__class__.__name__}(\n{self.v})'
    
    @property
    def shape(self):
        return self.v.shape
    
    @property
    def x(self):
        return self.v[2,:,:]
    
    @property
    def y(self):
        return self.v[1,:,:]
    
    @property
    def z(self):
        return self.v[0,:,:]
    
    @property
    def mag(self):
        return np.linalg.norm(self.v, axis =0) 
    
    @property
    def norm(self):
        v = self.v
        return self.from_array(v/np.linalg.norm(v, axis =0))
    
    
    @staticmethod
    def from_array(array):
        return Vector(array[0,:,:],array[1,:,:],array[2,:,:])
    
    def to_array(self):
        return self.v
    
    def to_size_like(self, other):
        '''Take a single element Vector and transform it to a 2D Vector of the same size of other '''
        vec = self.v
        o = np.ones_like(other.v[0])
        # self.v = np.array([vec[0]*o, vec[1]*o, vec[2]*o])
        return Vector(vec[0]*o, vec[1]*o, vec[2]*o)
        
        
    def zeros_like(self):
        i = self.v.shape[1]
        j = self.v.shape[2]
        xyz = np.zeros([i,j])
        return Vector(xyz,xyz,xyz)
    
    def ones_like(self):
        i = self.v.shape[1]
        j = self.v.shape[2]
        xyz = np.ones([i,j])
        return Vector(xyz,xyz,xyz)

    def __neg__(self):
        return self.from_array(-self.v)

    @extend_to_Vector
    def __add__(self, other):
        pass
    
    @extend_to_Vector
    def __sub__(self,other):
        pass
    
    @extend_to_Vector
    def __mul__(self,other):
        pass
    
    @extend_to_Vector
    def __truediv__(self,other):
        pass
    
    @extend_to_Vector
    def __divmod__(self,other):
        pass
    
    def cross(self,other):
        c = np.cross(self.v, other.v, axis=0)
        return self.from_array(c)
    
    def dot(self,other):
        return np.sum(self.v*other.v,axis=0)
    
if __name__ == '__main__':

    a = np.random.randint(0,9,[2,2]) 
    v = Vector(a,a,a)
    # w = Vector.from_array(np.random.randint(0,9,[3,2,2]))
    w = Vector.from_array(3*np.ones([3,2,2]))
    r = 3*np.ones([2,2])
    r[1,1] = 0
    print(v.shape)