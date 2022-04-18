from tablite2.utils import intercept


def test_range_intercept():
    A = range(500,700,3)
    B = range(520,700,3)
    C = range(10,1000,30)

    assert intercept(A,C) == range(0)
    assert set(intercept(B,C)) == set(B).intersection(set(C))

    A = range(500_000, 700_000, 1)
    B = range(10, 10_000_000, 1000)

    assert set(intercept(A,B)) == set(A).intersection(set(B))

    A = range(500_000, 700_000, 1)
    B = range(10, 10_000_000, 1)

    assert set(intercept(A,B)) == set(A).intersection(set(B))


if __name__ == "__main__":
    test_range_intercept()