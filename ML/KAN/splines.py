import torch
import torch.nn as nn
import torch.nn.functional as F

class CubicSplineFunction(nn.Module):
    def __init__(self, num_knots=20, x_range=(-5, 5)):
        super().__init__()
        self.num_knots = num_knots
        self.x_min, self.x_max = x_range

        # Learnable y-values at knots
        self.values = nn.Parameter(torch.randn(num_knots))  

        # Uniform knots (no need to save them explicitly)
        self.dx = (self.x_max - self.x_min) / (num_knots - 1)

    def forward(self, x):
        # Clamp input
        x = torch.clamp(x, self.x_min, self.x_max)

        # Normalize x to index space
        u = (x - self.x_min) / self.dx  # fractional index
        i = torch.floor(u).long()
        u = u - i  # local fractional coordinate in [0,1)

        # Get 4 control point indices for cubic B-spline
        i0 = torch.clamp(i - 1, 0, self.num_knots - 1)
        i1 = torch.clamp(i    , 0, self.num_knots - 1)
        i2 = torch.clamp(i + 1, 0, self.num_knots - 1)
        i3 = torch.clamp(i + 2, 0, self.num_knots - 1)

        # Fetch values at control points
        v0 = self.values[i0]
        v1 = self.values[i1]
        v2 = self.values[i2]
        v3 = self.values[i3]

        # Cubic B-spline basis (Catmull-Rom-like)
        u2 = u * u
        u3 = u2 * u

        # Basis matrix for Catmull-Rom (can be changed)
        out = 0.5 * (
            (-v0 + 3*v1 - 3*v2 + v3) * u3 +
            (2*v0 - 5*v1 + 4*v2 - v3) * u2 +
            (-v0 + v2) * u +
            2*v1
        )
        return out


class InterpolateFunction(nn.Module):
    def __init__(self, num_knots=20, x_range=(-5, 5)):
        super().__init__()
        self.num_knots = num_knots # number of knots to be interpolated
        self.x_min, self.x_max = x_range
        # For b-splines you would need knots but not here
        #self.knots = torch.linspace(self.x_min, self.x_max, num_knots)
        self.values = nn.Parameter(torch.randn(num_knots))  # Learnable spline values

    def forward(self, x):
        # Clamp and interpolate linearly between knots
        x_clamped = torch.clamp(x, self.x_min, self.x_max)
        x_norm = (x_clamped - self.x_min) / (self.x_max - self.x_min) * (self.num_knots - 1)
        #print(x.shape, x_norm.shape)
        x0 = torch.floor(x_norm).long()
        x1 = x0 + 1
        #print("------------",x0, x1) # left and right node to be interpolated between
        x1 = torch.clamp(x1, max=self.num_knots - 1)
        x0 = torch.clamp(x0, max=self.num_knots - 1)

        weight = x_norm - x0.float() # x_norm maybe 1.3 and x0=2 then the weight is 0.7
        v0 = self.values[x0]
        v1 = self.values[x1]
        return (1 - weight) * v0 + weight * v1  # Linear interpolation