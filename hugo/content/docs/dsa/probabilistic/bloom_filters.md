---
title: "Bloom Filters"
weight: 1
# bookFlatSection: false
# bookToc: true
# bookHidden: false
# bookCollapseSection: false
# bookComments: false
# bookSearchExclude: false
---

## probabilistic retrieval from a fuzzy set

[Bloom filters](https://en.wikipedia.org/wiki/Bloom_filter) are motivated by the following question: _how do we use a fixed amount of memory to check set-inclusion?_

As always there are tradeoffs. The tradeoff in this case is that we lose certainty --- but what in life is certain, anyway?

More formally, a bloom filter is a _probabilistic data structure_ --- a data structure that --- which implements the following operations:
* `insert(x)` --- inidicate that an item _x_ has been added into the set.
* `query(x)` --- check to see if an item _x_ has been inserted into the bloom filter previously.

Notably, the standard implementation of a Bloom filter does _not_ implement an operation to remove an element from the structure, since erasing an item's inclusion from the set would also corrupt the inclusion queries for other items.

## implementing a bloom filter with hash functions

How do we implement Bloom filters? The short answer is with _hash functions_ that map elements to addresses of a _memory bank_ of bits.

Imagine you have _m_ bits labeled 1 to _m_, and _k_ hash functions
$$
h_k: X \to \\{1,\ldots,m\\}
$$
so that there is little-to-no correlation between each hash function (this is important).

Then the way we `insert` an element _x_ is to compute all the hash values
$$
H(x) = \\{ h_1(x), \ldots, h_k(x) \\} \subset [m]
$$
and flip all of their bits from 0 to 1.

When we query for the existence of _x_ in the set represented by the Bloom filter, we compute _H(x)_ again, and check that all the bits are equal to 1; if there is even a single zero bit in the hashed bits _H(x)_, we return `false` for the query.

## false positive rates

It's possible to get a false positive with Bloom filters (the filter says _x_ is in the set, but it was never actually inserted).

It's not possible to get false negatives (the filter says _x_ is not in the set, but it's actually been inserted).

What's the chance of getting one of these false positives? Well, if we're using _k_ hash functions per element across _m_ bits, we can do some basic math to find that the probability of a collision after inserting _n_ elements is:
$$
\biggl( 1 - \exp\frac{-kn}{m} \biggr)^k
$$

## optimal settings for bloom filters

Say we have a fixed budget of $m$ elements in our memorybank. How many hash functions should we be using to minimize the false positive rate?

(Watch this space; this is a topic for a future update.)

## implementing it in python

In the below, we assume we can construct a set of hash functions based on big prime numbers and modular arithmetic; see [this link](https://www.cs.cornell.edu/courses/cs312/2008sp/lectures/lec21.html) for the complicated details of how that's possible. But in general, designing hash functions is _hard_, and it's a vibrant field of algorithmics research to design excellent hash functions for various purposes --- for instance, cryptographic security, low collision rates, et cetera.

{{< highlight python >}}
class BloomFilter(object):
  """
  Bloom filter implementation.
  """
  def __init__(self, numhashes, numbits):
	self.numhashes = numhashes
	self.numbits = numbits
    self.hash_functions = [ ]
	for k in range(numhashes):
		hk = (lambda x: COEFF[k]*x + OFFSET[k] % BIGPRIME[k])
		self.hash_functions.append(hk)
	self.memory = [0 for _ in range(numbits)]
	
  def insert(self, x):
    bits = [ hf(x) for hf in self.hash_functions ]
	for bit in bits:
		self.memory[bit] = 1
	
  def query(self, x):
    bits = [ hf(x) for hf in self.hash_functions ]
	return all([self.memory[b] == 1 for b in bits])
{{< / highlight >}}
