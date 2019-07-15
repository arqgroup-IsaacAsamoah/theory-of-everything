# The theory of everything

* next steps:
  - ~~Read about classification RFs (gini impurity)~~
  - ~~Write down key understandings of regression RFs~~
  - Implement regression RF
    - Integrate functional programming and pytest.
  - Write down key understandings of classification RFs
  - Implement classification RF
* Functional programming refresher
  - no side effects: uses nothing outside the function, affects nothing outside the function
  - recursion rather than changing state with loops inside a function
  - Higher order functions - take functions as arguments or return a function
  - look at recurse article for reference
  - comprehensions are the pyhton 3 equivalent pf map, reduce and filter etc
  - use lambda simple functions. pre-build functions as arguments for more complex ones.
  - Think about where I need to return a function from a function.
* Learned from Tom, zip is great to concatenate two columns of the same shape
* Also learned trees are a great place to try recursion.
* Need to revise the pluck function with fresh eyes.

My reflection so far is, it was too much to attempt to build a decision tree with
recursion without side effects all at once. I need to figure out how to build the tree (done) then how to build it with recursion, then see if I can do it without side effects.

new next steps :-)
* ~~create a bootstrap function - run it for each tree in the forest~~
* ~~test out my grow_forest with a list comprehension idea~~
* ~~write and test a predict function~~
* ~~test out by predict from forest list comprehension idea.~~
* write down algorithm now that I have actually implemented it.
* write up my learning journey
  - random forest theory intuition
  - random forest algorithm
  - random forest implementation
    - side topic: recursion
    - side topic: binary tree recursion
    - side topic functions without side effects
      - pure functions
      - immutable data
* learn about pytest - started
* set up tests for each of the functions I've written
* ~~use pytest to help me re-write recursive_feature_split without side effects~~
* add pytest to my RF learning journey
