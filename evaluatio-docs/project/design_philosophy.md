# Project and Design Philosophy

Evaluatio is more than just a fast metrics library, it is a project attempting to make statistically rigorous testing easier and more available to academics, researchers, hobbyists, and alike. This document is meant to serve as the foundation of the philosophy of the design of the library and project at large.  

### Making evaluations straightforward
Evaluation, statistical testing, and comparisons should not be difficult. While there are infinite ways of making mistakes when using statistics, the library and the documentation should make it how to use them properly and highlight common pitfalls.

### Strongly typed and type annotated
Ever tried to use libraries that have poor type annotation? Well, it sucks, and we aim to be better. Functions should only ever return one thing, and one thing only, and if it returns several values it should be in a struct with named fields.


### Not implement features for the sake of it
The core of the library uses Rust to optimise performance, but we will not implement metrics and functions without there being a legitimate reason. A legitimate reason might be performance, type-safety, compatibility, etc. but we will not reinvent the wheel for no reason.

### Encourage community improvement of evaluations
We are stronger together, and as such having a community verify and improve the ways we do evaluations makes us all better and ensures that our results are reproducible and trustworthy. Together we can make evaluations, statistical testing, and rigorous testing of models the norm.
