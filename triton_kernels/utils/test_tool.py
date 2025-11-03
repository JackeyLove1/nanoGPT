from tool import *

def test_tensor_cache_basic():
    """Test basic tensor_cache functionality: cache stores result"""
    call_count = 0

    @tensor_cache
    def compute_tensor(x: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return x * 2

    x = torch.tensor([1.0, 2.0, 3.0])
    result = compute_tensor(x)

    assert call_count == 1
    assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))


def test_tensor_cache_hit():
    """Test cache hit: same tensor object returns cached result without recomputation"""
    call_count = 0

    @tensor_cache
    def compute_tensor(x: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return x * 2

    x = torch.tensor([1.0, 2.0, 3.0])

    # First call
    result1 = compute_tensor(x)
    assert call_count == 1

    # Second call with same tensor object (cache hit)
    result2 = compute_tensor(x)
    assert call_count == 1  # Should NOT increment

    # Verify results are identical
    assert torch.equal(result1, result2)


def test_tensor_cache_miss_different_tensor():
    """Test cache miss: different tensor object triggers recomputation"""
    call_count = 0

    @tensor_cache
    def compute_tensor(x: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return x * 2

    x1 = torch.tensor([1.0, 2.0, 3.0])
    x2 = torch.tensor([1.0, 2.0, 3.0])  # Same values, different object

    result1 = compute_tensor(x1)
    assert call_count == 1

    result2 = compute_tensor(x2)
    assert call_count == 2  # Cache miss, function called again

    # Results are the same numerically but results are different objects
    assert torch.equal(result1, result2)


def test_tensor_cache_with_kwargs():
    """Test tensor_cache with keyword arguments"""
    call_count = 0

    @tensor_cache
    def compute_tensor(x: torch.Tensor, scale: int = 1) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return x * scale

    x = torch.tensor([1.0, 2.0, 3.0])

    # First call
    result1 = compute_tensor(x, scale=2)
    assert call_count == 1
    assert torch.allclose(result1, torch.tensor([2.0, 4.0, 6.0]))

    # Same args and kwargs -> cache hit
    result2 = compute_tensor(x, scale=2)
    assert call_count == 1

    # Different kwargs -> cache miss
    result3 = compute_tensor(x, scale=3)
    assert call_count == 2
    assert torch.allclose(result3, torch.tensor([3.0, 6.0, 9.0]))


def test_tensor_cache_multiple_args():
    """Test tensor_cache with multiple arguments"""
    call_count = 0

    @tensor_cache
    def compute_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return x + y

    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])

    # First call
    result1 = compute_tensor(x, y)
    assert call_count == 1

    # Same objects -> cache hit
    result2 = compute_tensor(x, y)
    assert call_count == 1

    # Different first argument
    x_new = torch.tensor([1.0, 2.0])
    result3 = compute_tensor(x_new, y)
    assert call_count == 2


def test_tensor_cache_preserves_metadata():
    """Test that @functools.wraps preserves function metadata"""

    @tensor_cache
    def my_special_function(x: torch.Tensor) -> torch.Tensor:
        """This is my special function"""
        return x * 2

    # Check function name is preserved
    assert my_special_function.__name__ == 'my_special_function'

    # Check docstring is preserved
    assert my_special_function.__doc__ == 'This is my special function'


def test_tensor_cache_no_initial_cache():
    """Test that the cache handles the first call when cache is empty"""
    call_count = 0

    @tensor_cache
    def compute_tensor(x: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call with initial empty cache
    x = torch.tensor([1.0, 2.0, 3.0])
    result = compute_tensor(x)

    assert call_count == 1
    assert result is not None


def test_tensor_cache_different_arg_lengths():
    """Test cache behavior with different argument lengths"""
    call_count = 0

    @tensor_cache
    def compute_tensors(*args: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return sum(args)

    x = torch.tensor([1.0])
    y = torch.tensor([2.0])
    z = torch.tensor([3.0])

    # Call with 2 args
    result1 = compute_tensors(x, y)
    assert call_count == 1

    # Call with 3 args -> different length, cache miss
    result2 = compute_tensors(x, y, z)
    assert call_count == 2

    # Call with 2 args again (same objects) -> but cache was updated, so cache hit for (x, y, z)
    # Actually this tests whether it's a cache miss because lengths differ
    result3 = compute_tensors(x, y)
    # Since we switched to (x,y,z), the cached state is now (x,y,z)
    # So calling with (x,y) will be a miss
    assert call_count == 3


def test_tensor_cache_non_tensor_args():
    """Test tensor_cache with mixed tensor and non-tensor arguments"""
    call_count = 0

    @tensor_cache
    def compute_tensor(x: torch.Tensor, scalar: float) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return x * scalar

    x = torch.tensor([1.0, 2.0, 3.0])

    # First call
    result1 = compute_tensor(x, 2.5)
    assert call_count == 1

    # Same args -> cache hit
    result2 = compute_tensor(x, 2.5)
    assert call_count == 1

    # Different scalar value -> cache miss
    result3 = compute_tensor(x, 3.5)
    assert call_count == 2


def test_tensor_cache_returns_correct_value():
    """Test that cache returns correct computed values"""

    @tensor_cache
    def square_tensor(x: torch.Tensor) -> torch.Tensor:
        return x ** 2

    x = torch.tensor([2.0, 3.0, 4.0])
    result = square_tensor(x)

    expected = torch.tensor([4.0, 9.0, 16.0])
    assert torch.allclose(result, expected)


def test_tensor_cache_identity_check():
    """Test that cache uses identity (is) not equality (==)"""
    call_count = 0

    @tensor_cache
    def identity_check(x: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return x.clone()

    # Create two tensors with same values but different objects
    x1 = torch.tensor([1.0, 2.0, 3.0])
    x2 = torch.tensor([1.0, 2.0, 3.0])

    assert torch.equal(x1, x2)  # Values are equal
    assert x1 is not x2  # But objects are different

    result1 = identity_check(x1)
    call_count_after_first = call_count

    result2 = identity_check(x2)

    # Since x1 and x2 are different objects, cache should miss
    assert call_count == call_count_after_first + 1


if __name__ == "__main__":
    # Run tests with: pytest fla/utils.py::test_tensor_cache_basic -v
    pass
