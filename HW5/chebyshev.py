# This code was written by Prof. Yosef Zlochower.

from numpy import sinc, sin, cos, pi, arccos, fabs
import numpy as np
from functools import cache

"""
Routines for working with Chebyshev polynomials.
"""

eps = 1.0e-10
# we want the unormalized sinc
def _Sinc(x):
    return sinc(x / pi) * pi


def ReconT(C, x):
    """
    Returns the value of a Chebyshev series at a point x given a
    set of Chebyshev coefficients 'C'.
    """

    ret = 0
    for n, c in enumerate(C):
        ret += c * ChebyshevT(n, x)
    return ret

def ReconTx(C, x):
    """
    Returns the value of a Chebyshev series at a point x given a
    set of Chebyshev coefficients 'C'.
    """

    ret = 0
    for n, c in enumerate(C):
        ret += c * ChebyshevTx(n, x)
    return ret

def ReconTxx(C, x):
    """
    Returns the value of a Chebyshev series at a point x given a
    set of Chebyshev coefficients 'C'.
    """

    ret = 0
    for n, c in enumerate(C):
        ret += c * ChebyshevTxx(n, x)
    return ret

@cache
def ChebyshevT(n, x):
    """Evaluates the Chebyshev polynomial of the first kind of order n
    and point x.
    """

    phi = arccos(x)
    return cos(n * phi)


@cache
def ChebyshevU(n, x):
    """Evaluates the Chebyshev polynomial of the second kind of order n
    and point x.
    """

    phi = arccos(x)
    s = 1
    if phi > 0.5 * pi:
        s = 1 if n % 2 == 0 else -1
        phi = pi - phi

    # re-written in terms of _Sinc to avoid 0/0 and associated loss of
    # precision
    return s * (n + 1) * _Sinc((n + 1) * phi) / _Sinc(phi)


@cache
def ChebyshevTx(n, x):
    """
    Evaluates the first derivative of the Chebyshev polynomial of the
    first kind of order n at point x.
    """

    if fabs(x * x - 1) < eps:
        s = 1 if n % 2 != 0 else x
        return n * n * s  # n^2 or (-1)^n n^2 if x == -1

    return n * ChebyshevU(n - 1, x)


@cache
def ChebyshevTxx(n, x):
    """
    Evaluates the second derivative of the Chebyshev polynomial of the
    first kind of order n at point x.
    """

    if fabs(x * x - 1) < eps:
        s = 1 if n % 2 == 0 else x
        return n * n * (n * n - 1) / 3.0 * s
    return n * (n * ChebyshevT(n, x) - x * ChebyshevU(n - 1, x)) / (x * x - 1)


def RootsGrid(nmax):
    """
    Generates an array containing the collocation points corresponding
    to the roots of the nmax+1 Chebyshev polynomial of the first kind.
    """

    grid = np.ndarray(shape=(nmax + 1), dtype=np.float64)
    for i in range(nmax + 1):
        grid[i] = cos(pi * ((nmax - i) + 0.5) / (nmax + 1))
    return grid


def GaussLobattoGrid(nmax):
    """
    Generates an array containing the collocation points corresponding
    to the Gauss-Lobatto points.
    """

    grid = np.ndarray(shape=(nmax + 1), dtype=np.float64)
    for i in range(nmax + 1):
        grid[i] = cos(pi * (nmax - i) / nmax)
    return grid


def Chebyshev_RootsGrid_to_Coefficient(nmax, gridx, gridval):
    """
    Given a set of function values on the RootsGrid, gives the
    associated Chebyshev coefficients for an expansion in Chebyshev
    polynomials of the first kind.
    """

    vals = np.ndarray(shape=(nmax + 1), dtype=np.float64)

    norm = 1.0 / (nmax + 1.0)
    for m in range(nmax + 1):
        vals[m] = 0
        for l in range(nmax + 1):
            vals[m] += ChebyshevT(m, gridx[l]) * gridval[l]

        vals[m] *= norm
        norm = 2.0 / (nmax + 1.0)
    return vals


def Chebyshev_Coefficients_to_Grid(nmax, gridx, coefs):
    """
    Given a set of coefficients for an expansion using Chebyshev
    polynomials of the first kind, and a grid of coordinates, returns
    the values of the series on that grid.
    """

    vals = np.ndarray(shape=(nmax + 1), dtype=np.float64)
    for l in range(nmax + 1):
        vals[l] = 0
        for m in range(nmax + 1):
            vals[l] += ChebyshevT(m, gridx[l]) * coefs[m]
    return vals
