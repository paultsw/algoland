+++
title = 'Data Structures and Algorithms'
draft = false
+++

## algorithms

_Algorithms_ are recipes for your computer. They are recipes, usually implemented on a computer following a Turing machine (or a Von Neumann architectured machine), but can also refer to those that are done on an abstract lambda calculus.

In particular, algorithms are usually clever recipes implemented to solve a particular problem. This can involve things like: how to count the number of ways to make change for a certain number of dollars; how to find the fastest route from start to finish in a given topology; or how to arrange a set of jumbled numbers into an orderly sequence.

## data structures

_Data structures_, on the other hand, are abstract structures that allow us to store, retrieve, and manipulate data in convenient ways.

What we mean by "convenient" can vary --- maybe it means we can do something quickly. Maybe it means we can yank out a certain element of the dataset quickly. Or maybe it means we can add something to a collection quickly.

Data structures are closely related to algorithms. For instance, we might need a clever algorithm to set up a data structure in a meaningfully quick way. Or maybe a data structure relies on a clever algorithm to retrieve or modify an existing data element.

Data structures come in many many flavors:
* maybe we're dealing with a _tree_, which is a data structure that has a hierarchical quality to it; this can enable, for instance, fast lookups if we have a balanced binary tree.
* _graphs_ are data structures and can be represented on a computer in many ways; they represent datasets where the individual elements have some sort of connection with other elements.
* We have _randomized_, or _probabilistic data structures: structures that don't offer a hard-and-fast guarantee of good performance, but guarantee good performance "almost all the time", by using randomness in the underlying methods.
* Another popular structure is the _hash table_, or just _table_. These rely on the properties of certain mathematical functions, called _hash functions_, to provide fast lookups.
