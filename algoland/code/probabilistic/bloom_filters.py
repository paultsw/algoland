"""
A bloom filter BF is a probabilistic data structure that supports the following operations:
* insert(BF, x) -- indicate that 'x' is contained in the bloom filter.
* isin(BF, x) -- reports whether or not 'x' has been previously inserted into the BF.

In other words, a BF is a set-like structure that lets you put items into it and query
for existence.

The caveat is that the `isin` function is only probabilistically correct: i.e. it is
possible for there to be false positives: `isin` returning True for an `x` that was never
previously inserted before.

On the plus side, though, you can instantiate a BF with a fixed amount of memory, specified
at construction-time. For example, you can instantiate it with M bits, and the set structure
will support insertion into the BF without increasing the number of bits.

How does it work? Well, you also instantiate it with a bank of K hash functions, each of which
maps along the M bits.
"""
# List of first 100 primes; we use this to build our hash functions below.
# Note that there's no need to rely on our specific hash function implementation;
# anything that is evenly-distributed on [0,...,m-1] would work.
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
    229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
    313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401,
    409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491,
    499, 503, 509, 521, 523, 541
]
BIGPRIME = 695401127868109045639800373663
def create_hash_functions(k,m):
    """
    Create K different hash functions, each of which maps a string
    to an integer from 0 to m-1. Returns a list of lambda functions.
    """
    hash_fns = []
    for it in range(k):
        hash_fns.append(
            lambda st: ((sum([ord(ch) for ch in st] * BIGPRIME)) % PRIMES[it]) % m
        )
    return hash_fns


class BloomFilter(object):
    def __init__(self, bitarray_size, hash_bank_size):
        self.bitarray_size = bitarray_size
        self.hash_bank_size = hash_bank_size
        self._hash_fns = create_hash_functions(hash_bank_size, bitarray_size)
        self._bitarray = [ False for _ in range(bitarray_size) ]

    def insert(self, key):
        for k in range(self.hash_bank_size):
            self._bitarray[self._hash_fns[k](key)] = True

    def query(self, key):
        res = [ self._bitarray[self._hash_fns[k](key)] for k in range(self.hash_bank_size) ]
        return all(res)


def bloom_filter_usage_example():
    """
    Example of usage of our Bloom Filter class.
    """
    # construct a bloom filter with a bit-array of size 1000 and
    # 100 hash functions in the bank:
    BANKSIZE, ARRAYSIZE = 100, 1000
    bf = BloomFilter(ARRAYSIZE, BANKSIZE)

    # insert a string into the bloom filter:
    bf.insert("hello world!")
    
    # guaranteed to return True (i.e., no false negatives):
    print( bf.query("hello world!") )

    # should return False with high probability; could return True with low probability:
    print( bf.query("!dlrow olleh") )
