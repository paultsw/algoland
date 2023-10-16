"""
Basic code templates from LeetCode.

The purpose of this file is basically auto-didactic: you'll never be able to fully grok advanced algorithms
if you don't have the absolute basic templates down, so for each template below, we recommend rewriting them
from scratch at least 10x to boost muscle-memory. Try to do those in the `practice rewriting the above` section
below each template.
"""

##########
# TWO POINTERS -- ONE INPUT, OPPOSITE ENDS
#
def fn(arr):
    left = ans = 0
    right = len(arr) - 1

    while left < right:
        # do some logic here with left and right
        if CONDITION:
            left += 1
        else:
            right -= 1
    
    return ans

# ----- practice rewriting the above:
def fn(arr):
    left = ans = 0
    right = len(arr) - 1
    while left < right:
        # DO SOMETHING HERE WITH L & R
        if CONDITION:
            left += 1
        else:
            right -= 1
    return ans

def fn(arr):
    left = ans = 0
    right = len(arr) - 1
    while left < right:
        # DO SOMETHING HERE WITH LEFT AND RIGHT
        if CONDITION:
            left += 1
        else:
            right -= 1
    return ans

def fn(arr):
    left = ans = 0
    right = len(arr) - 1
    while left < right:
        # DO SOMETHING WITH L & R HERE
        if CONDITION:
            left += 1
        else:
            right -= 1
    return ans

##########
# TWO POINTERS -- TWO INPUTS, EXHAUST BOTH
#
def fn(arr1, arr2):
    i = j = ans = 0

    while i < len(arr1) and j < len(arr2):
        # do some logic here
        if CONDITION:
            i += 1
        else:
            j += 1
    
    while i < len(arr1):
        # do logic
        i += 1
    
    while j < len(arr2):
        # do logic
        j += 1
    
    return ans

# ----- practice rewriting the above:
def fn(arr1, arr2):
    i = j = ans = 0
    while i < len(arr1) and j < len(arr2):
        # DO SOME LOGIC HERE
        if CONDITION:
            i += 1
        else:
            j += 1

    # clean up arr1:
    while i < len(arr1):
        # DO LOGIC HERE
        i += 1

    # clean up arr2:
    while j < len(arr2):
        # DO LOGIC HERE
        j += 1

    return ans


def fn(arr1, arr2):
    i = j = ans = 0
    while i < len(arr1) and j < len(arr2):
        # DO SOME LOGIC HERE
        if CONDITION:
            i += 1
        else:
            j += 1
    # clean up arr1:
    while i < len(arr1):
        # DO LOGIC HERE
        i += 1
    # clean up arr2:
    while j < len(arr2):
        j += 1

    return ans
        
##########
# SLIDING WINDOW
#
def fn(arr):
    left = ans = curr = 0

    for right in range(len(arr)):
        # do logic here to add arr[right] to curr

        while WINDOW_CONDITION_BROKEN:
            # remove arr[left] from curr
            left += 1

        # update ans
    
    return ans


# ----- practice rewriting the above:
def fn(arr):
    left = ans = curr = 0
    for right in range(len(arr)):
        # DO LOGIC HERE; ADD arr[right] to curr somehow
        while WINDOW_CONDITION_BROKEN:
            # REMOVE arr[left] FROM curr HERE
            left += 1
        # UPDATE ANSWER HERE
    return ans

def fn(arr):
    left = curr = ans = 0
    for right in range(len(arr)):
        # DO LOGIC HERE: ADD arr[right] to curr somehow
        while WINDOW_CONDITION_BROKEN:
            # remove arr[left] FROM curr HERE
            left += 1
        # UPDATE ans HERE
    return ans

##########
# BUILDING A PREFIX SUM
#
def fn(arr):
    prefix = [arr[0]]
    for i in range(1, len(arr)):
        prefix.append(prefix[-1] + arr[i])
    
    return prefix


# ----- practice rewriting the above:
def fn(arr):
    prefixes = [ arr[0] ]
    for i in range(1, len(arr)):
        prefix.append( prefix[-1]+arr[i] )
    return prefixes

def fn(arr):
    prefixes  = [ arr[0] ]
    for i in range(1, len(arr)):
        prefix.append( prefix[-1] + arr[i] )
    return prefixes

##########
# EFFICIENT STRING BUILDING
# arr is a list of characters
def fn(arr):
    ans = []
    for c in arr:
        ans.append(c)
    
    return "".join(ans)


# ----- practice rewriting the above:
def fn(arr):
    ans = []
    for c in arr:
        ans.append(c)
    return "".join(ans)

def fn(arr):
    ans = []
    for c in arr:
        ans.append(c)
    return "".join(ans)

##########
# LINKED LIST -- FAST AND SLOW POINTERS
# 
def fn(head):
    slow = head
    fast = head
    ans = 0

    while fast and fast.next:
        # do logic
        slow = slow.next
        fast = fast.next.next
    
    return ans

# ----- practice rewriting the above:
def fn(head):
    slow = head
    fast = head
    ans = 0

    while fast and fast.next:
        # DO LOGIC HERE
        slow = slow.next
        fast = fast.next.next

    return ans

##########
# REVERSING A LINKED LIST
# 
def fn(head):
    curr = head
    prev = None
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node 
        
    return prev


# ----- practice rewriting the above:
def fn(head):
    curr = head
    prev = None
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

##########
# FIND NUMBER OF SUBARRAYS FITTING AN EXACT CRITERION
# 
from collections import defaultdict

def fn(arr, k):
    counts = defaultdict(int)
    counts[0] = 1
    ans = curr = 0

    for num in arr:
        # do logic to change curr
        ans += counts[curr - k]
        counts[curr] += 1
    
    return ans


# ----- practice rewriting the above:
def fn(arr, k):
    counts = defaultdict(int)
    counts[0] = 1
    ans = curr = 0
    for num in arr:
        # DO LOGIC TO CHANGE curr HERE
        ans += counts[curr - k]
        counts[curr] += 1
    return ans

##########
# MONOTONIC INCREASING STACK
# 
def fn(arr):
    stack = []
    ans = 0

    for num in arr:
        # for monotonic decreasing, just flip the > to <
        while stack and stack[-1] > num:
            # do logic
            stack.pop()
        stack.append(num)
    
    return ans


# ----- practice rewriting the above:
def fn(arr):
    stack = []
    ans = 0
    for num in arr:
        # NOTE: for monotonic *decreasing*, just flip > to <
        while stack and stack[-1] > num:
            # DO LOGIC HERE
            stack.pop()
        stack.append(num)
    return ans

##########
# BINARY TREE -- DFS (Recursive)
# 
def dfs(root):
    if not root:
        return
    
    ans = 0

    # do logic
    dfs(root.left)
    dfs(root.right)
    return ans


# ----- practice rewriting the above:
def dfs(root):
    # base:
    if not root:
        return

    ans = 0
    # DO LOGIC HERE TO `ans`

    # recurse:
    dfs(root.left)
    dfs(root.right)

    return ans

##########
# BINARY TREE -- DFS (Iterative)
# 
def dfs(root):
    stack = [root]
    ans = 0

    while stack:
        node = stack.pop()
        # do logic
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return ans


# ----- practice rewriting the above:
def dfs(root):
    stack = [root]
    ans = 0
    while stack:
        node = stack.pop()
        # DO LOGIC HERE
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return ans

##########
# BINARY TREE -- BFS
# 
from collections import deque

def fn(root):
    queue = deque([root])
    ans = 0

    while queue:
        current_length = len(queue)
        # do logic for current level

        for _ in range(current_length):
            node = queue.popleft()
            # do logic
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return ans


# ----- practice rewriting the above:
def fn(root):
    queue = deque([root])
    ans = 0

    while queue:
        current_length = len(queue)
        # DO LOGIC HERE FOR CURRENT LEVEL
        for _ in range(current_length):
            node = queue.popleft()
            # DO LOGIC HERE ON NODE
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return ans

##########
# GRAPH -- DFS (Recursive)
# 
def fn(graph):
    def dfs(node):
        ans = 0
        # do some logic
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                ans += dfs(neighbor)
        
        return ans

    seen = {START_NODE}
    return dfs(START_NODE)


# ----- practice rewriting the above:
def fn(graph):
    def dfs(node):
        ans = 0
        # DO SOME LOGIC HERE
        for nbr in graph[node]:
            if nbr not in seen:
                seen.add(nbr)
                ans += dfs(nbr)
        return ans

    seen = {START_NODE}
    return dfs(START_NODE)

##########
# GRAPH -- DFS (Iterative)
# 
def fn(graph):
    stack = [START_NODE]
    seen = {START_NODE}
    ans = 0

    while stack:
        node = stack.pop()
        # do some logic
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    
    return ans

# ----- practice rewriting the above:
def fn(graph):
    stack = [START_NODE]
    seen = {START_NODE]
    ans = 0
    while stack:
        node = stack.pop()
        # DO SOME LOGIC HERE
        for nbr in graph[node]:
            if nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return ans

def fn(graph):
    stack = [START_NODE]
    seen = {START_NODE}
    ans = 0
    while stack:
        node = stack.pop()
        # DO SOME LOGIC HERE
        for nbr in graph[node]:
            if nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return ans
##########
# GRAPH -- BFS
# 
from collections import deque

def fn(graph):
    queue = deque([START_NODE])
    seen = {START_NODE}
    ans = 0

    while queue:
        node = queue.popleft()
        # do some logic to update `ans`
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    
    return ans


# ----- practice rewriting the above:
def fn(graph):
    queue = deque([START_NODE])
    seen = {START_NODE}
    ans = 0
    while queue:
        node = queue.popleft()
        # DO SOME LOGIC TO UPDATE ANSWER HERE
        for nbr in graph[node]:
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return ans

def fn(graph):
    queue = deque([START_NODE])
    seen = {START_NODE}
    ans = 0
    while queue:
        node = queue.popleft()
        # DO SOME LOGIC ON NODE TO UPDATE ANSWER HERE
        for nbr in graph[node]:
            if nbr not in seen:
                queue.append(nbr)
                seen.add(nbr)
    return ans

##########
# HEAP -- FIND TOP-K ELEMENTS
# 
import heapq

def fn(arr, k):
    heap = []
    for num in arr:
        # do some logic to push onto heap according to problem's criteria
        heapq.heappush(heap, (CRITERIA, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for num in heap]

# ----- practice rewriting the above:
def fn(arr, k):
    heap = []
    for num in arr:
        # DO LOGIC TO PUSH ONTO HEAP (DEPENDING ON PROBLEM)
        heapq.heappush(heap, (HEAP_CRITERIA, num))
        if len(heap) > k:
            heapq.heappop(heap)
    return [num for num in heap]

def fn(arr, k):
    heap = []
    for num in arr:
        # DO LOGIC TO PUSH ONTO HEAP (DEPENDING ON PROBLEM)
        heapq.heappush(heap, (HEAP_CRITERIA, num))
        if len(heap) > k:
            heapq.heappop(heap)
    return [num for num in heap]

##########
# BINARY SEARCH -- BASIC
# 
def fn(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            # do something
            return
        if arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    
    # left is the insertion point
    return left


# ----- practice rewriting the above:
def fn(arr, target):
    left = 0
    right = len(arr)-1
    while left <= right:
        mid = (left+right)//2
        if arr[mid] == target:
            # FOUND; DO SOMETHING
            return
        if arr[mid] > target:
            # search left half:
            right = mid-1
        else:
            # search right half
            left = mid+1
    # insertion point is on the left:
    return left

def fn(arr, target):
    left = 0
    right = len(arr)-1
    while left <= right:
        mid = (left+right)//2
        if arr[mid] == target:
            # FOUND; DO SOMETHING
            return
        if arr[mid] > target:
            # search left half:
            right = mid-1
        else:
            # search right half:
            left = mid+1
    # insertion point is on `left`:
    return left

##########
# BINARY SEARCH -- DUPLICATE ELEMENTS, LEFT-MOST INSERTION POINT
# 
def fn(arr, target):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] >= target:
            right = mid
        else:
            left = mid + 1

    return left


# ----- practice rewriting the above:
def fn(arr, target):
    left = 0
    right = len(arr)-1
    while left < right:
        mid = (left+right)//2
        if arr[mid] >= target:
            right = mid
        else:
            left = mid+1
    return left


##########
# BINARY SEARCH -- DUPLICATE ELEMENTS, RIGHT-MOST INSERTION POINT
# 
def fn(arr, target):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > target:
            right = mid
        else:
            left = mid + 1

    return left


# ----- practice rewriting the above:
def fn(arr, target):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left+right)//2
        if arr[mid] > target:
            right = mid
        else:
            left = mid+1
    return left

##########
# BINARY SEARCH FOR GREEDY PROBLEMS -- LOOKING FOR MINIMUM
# 
def fn(arr):
    def check(x):
        # this function is implemented depending on the problem
        return BOOLEAN

    left = MINIMUM_POSSIBLE_ANSWER
    right = MAXIMUM_POSSIBLE_ANSWER
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            right = mid - 1
        else:
            left = mid + 1
    
    return left


# ----- practice rewriting the above:
def fn(arr):
    def check(x):
        # THIS FUNCTION DEPENDS ON PROBLEM/USECASE
        return BOOLEAN

    left = MINIMUM_POSSIBLE_ANSWER
    right = MAXIMUM_POSSIBLE_ANSWER
    while left <= right:
        mid = (left+right)//2
        if check(mid):
            right = mid-1
        else:
            left = mid+1

    return left

##########
# BINARY SEARCH FOR GREEDY PROBLEMS -- LOOKING FOR MAXIMUM
# 
def fn(arr):
    def check(x):
        # this function is implemented depending on the problem
        return BOOLEAN

    left = MINIMUM_POSSIBLE_ANSWER
    right = MAXIMUM_POSSIBLE_ANSWER
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            left = mid + 1
        else:
            right = mid - 1
    
    return right


# ----- practice rewriting the above:
def fn(arr):
    def check(x):
        # THIS FUNCTION DEPENDS ON USE CASE
        return BOOLEAN

    left = MINIMUM_POSSIBLE_ANSWER
    right = MAXIMUM_POSSIBLE_ANSWER

    while left <= right:
        mid = (left+right)//2
        if check(mid):
            left = mid+1
        else:
            right = mid-1
    return right


##########
# BACKTRACKING 1
# Basic backtracking template. For a more advanced template,
# see the Skiena template below.
def backtrack(curr, OTHER_ARGUMENTS...):
    if (BASE_CASE):
        # modify the answer
        return
    
    ans = 0
    for (ITERATE_OVER_INPUT):
        # modify the current state
        ans += backtrack(curr, OTHER_ARGUMENTS...)
        # undo the modification of the current state
    
    return ans


# ----- practice rewriting the above:
def backtrack(curr, OTHER_ARGUMENTS...):
    if (BASE_CASE):
        # MODIFY THE ANSWER HERE
        return

    ans = 0
    for (ITERATE_OVER_INPUT):
        # MODIFY THE CURR STATE HERE
        ans += backtrack(curr, OTHER_ARGUMENTS...)
        # UNDO MODIFICATION HERE

    return ans


def backtrack(curr, OTHER_ARGUMENTS...):
    if (BASE_CASE):
        # MODIFY THE ANSWER HERE
        return

    ans = 0
    for (ITERATE_OVER_INPUTS):
        # MODIFY CURR STATE HERE
        ans += backtrack(curr, OTHER_ARGUMENTS...)
        # UNDO MODIFICATION HERE

    return ans


##########
# BACKTRACKING 2 (SKIENA)
# The backtracking template adapted from Skiena, "Algorithm Design Manual".
def is_solution(A,state):
    # TODO: check if 'A' forms one potential solution to the problem.
    # This is a separate function because sometimes this step can be a bit
    # involved.
    pass

def process_solution(A,state,solutions):
    # TODO: typically, append 'A' to 'solutions' container.
    # This is a separate func because sometimes 'A' needs to be processed.
    pass

def get_candidates(A,state,all_candidates):
    # TODO: filter all_candidates based on 'A' and 'state' here
    # and return possible next candidates for appending to 'A'.
    pass

def backtrack(A,state,all_candidates,solutions):
    if is_solution(A,state):
        process_solution(A,state,solutions)
    else:
        state = UPDATE_STATE(state) # TODO: update state here
        for cand in get_candidates(A,state,all_candidates):
            A.append(cand) # Add 'cand' to 'A' somehow
            backtrack(A,state,all_candidates,solutions)
            A.pop(cand) # remove 'cand' and move back up the state tree

# ----- practice rewriting the above:
def is_solution(A,state):
    pass

def process_solution(A,state,solutions):
    pass

def get_candidates(A,state,all_candidates):
    pass

def backtrack(A,state,all_candidates,solutions):
    if is_solution(A,state):
        process_solution(A,state,solutions)
    else:
        state = UPDATE_STATE(state) # UPDATE HERE
        for cand in get_candidates(A,state,all_candidates):
            A.append(cand)
            backtrack(A,state,all_candidates,solutions)
            A.pop(cand)


##########
# DYNAMIC PROGRAMMING -- TOP-DOWN MEMOIZATION
# 
def fn(arr):
    def dp(STATE):
        if BASE_CASE:
            return 0
        
        if STATE in memo:
            return memo[STATE]
        
        ans = RECURRENCE_RELATION(STATE)
        memo[STATE] = ans
        return ans

    memo = {}
    return dp(STATE_FOR_WHOLE_INPUT)


# ----- practice rewriting the above:
def fn(arr):
    # memoized function:
    def dp(STATE):
        if BASE_CASE:
            return 0
        if STATE in memo:
            return memo[STATE]
        ans = RECURRENCE_RELATION(STATE)
        memo[STATE] = ans
        return ans
    # construct memoization structure and run DP
    memo = {}
    return dp(STATE_FOR_WHOLE_INPUT)


def fn(arr):
    # memoized function:
    def dp(STATE):
        if BASE_CASE:
            return 0
        if STATE in memo:
            return memo[STATE]
        ans = RECURRENCE_RELATION(STATE)
        memo[STATE] = ans
        return ans
    # construct memoization structure and run DP:
    memo = {}
    return dp(STATE_FOR_WHOLE_INPUT)

##########
# BUILDING A TRIE
# note: using a class is only necessary if you want to store data at each node.
# otherwise, you can implement a trie using only hash maps.
# 
class TrieNode:
    def __init__(self):
        # you can store data at nodes if you wish
        self.data = None
        self.children = {}

def fn(words):
    root = TrieNode()
    for word in words:
        curr = root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        # at this point, you have a full word at curr
        # you can perform more logic here to give curr an attribute if you want
    
    return root


# ----- practice rewriting the above:
class TrieNode:
    def __init__(self):
        # (can store data within nodes if necessary)
        self.data = None
        self.children = {}

def fn(words):
    root = TrieNode()
    for word in words:
        curr = root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        # NOTE: at this point, have full word within curr
        # Can perform more logic here to give curr an attribute if necessary
    return root


class TrieNode:
    def __init__(self):
        self.data = None
        self.children = {}

def fn(words):
    root = TrieNode()
    # outer loop: insert all words
    for word in words:
        curr = root
        # inner loop: insert all chars in word
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        # NOTE: at this point have the full word inside curr.
        # Can perform more logic here to give curr an attribute if necessary
    return root

##########
# DIJKSTRA'S ALGORITHM
# 
from math import inf
from heapq import *

distances = [inf] * n
distances[source] = 0
heap = [(0, source)]

while heap:
    curr_dist, node = heappop(heap)
    if curr_dist > distances[node]:
        continue
    
    for nei, weight in graph[node]:
        dist = curr_dist + weight
        if dist < distances[nei]:
            distances[nei] = dist
            heappush(heap, (dist, nei))


# ----- practice rewriting the above:
from math import inf
from heapq import *

def dijkstra(graph,source,n):
    distances = [inf] * n
    distances[source] = 0
    heap = [(0,source)]

    while heap:
        curr_dist, node = heappop(heap)
        if curr_dist > distances[node]:
            continue

        for nbr, weight in graph[node]:
            dist = curr_dist + weight
            if dist < distances[nbr]:
                distances[nbr] = dist
                heappush(heap, (dist,nbr))

    return distances


def dijkstra(graph,source,n):
    distances = [inf] * n
    distances[source] = 0
    heap = [(0,source)]

    while heap:
        curr_dist, node = heappop(node)
        if curr_dist > distances[node]:
            continue

        for nbr, weight in graph[node]:
            new_dist = curr_dist + weight
            if new_dist < distances[nbr]:
                distances[nbr] = new_dist
                heappush(heap, (new_dist,nbr))

    return distances


def dijkstra(graph,source,n):
    distances = [inf] * n
    distances[source] = 0
    heap = [(0,source)]

    while heap:
        curr_dist, node = heappop(node)
        if curr_dist > distances[node]:
            continue

        for nbr, weight in graph[node]:
            new_dist = curr_dist + weight
            if new_dist < distances[nbr]:
                distances[nbr] = new_dist
                heappush(heap, (new_dist,nbr))

    return distances


##########
# UNION-FIND DATA STRUCTURE
# 


##########
# KRUSKAL'S ALGORITHM (MST)
# 
def kruskals_mst(self):
    # Resulting tree
    result = []
    
    # Iterator
    i = 0
    # Number of edges in MST
    e = 0

    # Sort edges by their weight
    self.m_graph = sorted(self.m_graph, key=lambda item: item[2])
    
    # Auxiliary arrays
    parent = []
    subtree_sizes = []

    # Initialize `parent` and `subtree_sizes` arrays
    for node in range(self.m_num_of_nodes):
        parent.append(node)
        subtree_sizes.append(0)

    # Important property of MSTs
    # number of egdes in a MST is 
    # equal to (m_num_of_nodes - 1)
    while e < (self.m_num_of_nodes - 1):
        # Pick an edge with the minimal weight
        node1, node2, weight = self.m_graph[i]
        i = i + 1

        x = self.find_subtree(parent, node1)
        y = self.find_subtree(parent, node2)

        if x != y:
            e = e + 1
            result.append([node1, node2, weight])
            self.connect_subtrees(parent, subtree_sizes, x, y)
    
    # Print the resulting MST
    for node1, node2, weight in result:
        print("%d - %d: %d" % (node1, node2, weight))


##########
# PRIM'S ALGORITHM (MST)
# 
def prims(graph):
    # used to pick minimum weight edge
    key = [self.INF for _ in range(self.V)]
    # used to store MST
    parent = [None for _ in range(self.V)]
    # pick a random vertex, ie 0
    key[0] = 0
    # create list for t/f if a node is connected to the MST
    mstSet = [False for _ in range(self.V)]
     # set the first node to the root (ie have a parent of -1)
    parent[0] = -1

    for _ in range(self.V):
        # 1) pick the minimum distance vertex from the current key
        # from the set of points not yet in the MST
        u = self.minKey(key, mstSet)
        # 2) add the new vertex to the MST
        mstSet[u] = True

        # loop through the vertices to update the ones that are still
        # not in the MST
        for v in range(self.V):
            # if the edge from the newly added vertex (v) exists,
            # the vertex hasn't been added to the MST, and
            # the new vertex's distance to the graph is greater than the distance
            # stored in the initial graph, update the "key" value to the
            # distance initially given and update the parent of
            # of the vertex (v) to the newly added vertex (u)
            if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                key[v] = self.graph[u][v]
                parent[v] = u

