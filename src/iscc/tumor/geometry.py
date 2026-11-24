"""Lattice geometry for the ductal field: the ring of a gland and the demes inside it."""
import pandas as pd


def bresenham_circumference(x0, y0, radius):
    x = radius
    y = 0
    err = 0

    points = []

    while x >= y:
        points.append((x0 + x, y0 + y))
        points.append((x0 + y, y0 + x))
        points.append((x0 - y, y0 + x))
        points.append((x0 - x, y0 + y))
        points.append((x0 - x, y0 - y))
        points.append((x0 - y, y0 - x))
        points.append((x0 + y, y0 - x))
        points.append((x0 + x, y0 - y))

        y += 1
        err += 1 + 2*y
        if 2*(err-x) + 1 > 0:
            x -= 1
            err += 1 - 2*x

    points = pd.DataFrame(points).drop_duplicates().values
    points = [(p[0], p[1]) for p in points]
    return points


def get_inside(border):
    points = []
    for (x,y) in border:
        # Add points inside -- general for any shape
        # find another x with this y
        for b in border:
            if b[1] == y:
                if b[0]-x > 1: # if to the right
                    points.extend([(x_,y) for x_ in range(x+1,b[0])])
                if b[0]-x > 1: # if to the left
                    points.extend([(x_,y) for x_ in range(b[0],x-1)])
    points = pd.DataFrame(points).drop_duplicates().values
    points = [(p[0], p[1]) for p in points]
    return points
