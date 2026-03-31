# Project and Design Philosophy

Evaluatio is more than just a fast metrics library, it is a project library for statistically rigorous evaluation.

The goal of the project is simple: make correct evaluation the default, not the exception. Too often, model evaluation is inconsistent, statistically unsound, or difficult to reproduce. Evaluatio aims to reduce these issues by providing well-defined metrics, principled statistical tools, and clear guidance on how to use them.

Statistically rigorous evaluation should be available to academics, researchers, hobbyists, and alike.

### Make correct evaluation easy and incorrect evaluation harder
Evaluation and statistical testing should be accessible, but not at the cost of correctness.

Evaluatio is designed to:
- make common evaluation workflows straightforward
- guide users toward statistically sound practices
- highlight common pitfalls and invalid comparisons

### Strong typing and explicit interfaces
Clear interfaces lead to more reliable results.

Evaluatio emphasises:
- strong type annotations
- predictable return types
- structured outputs with named fields

This reduces ambiguity, improves readability and developer experience, and helps catch errors early.

### Purpose-driven design
Features are not added unless they serve a clear purpose.

A new metric or functionality should:
- solve a real problem
- improve correctness, performance, or usability
- integrate cleanly with the rest of the library

Evaluatio does not aim to reimplement existing tools without a clear benefit. In many circumstances, sign-posting users to existing tools is preferable.

### Reproducibility and shared standards
Reliable evaluation requires shared understanding.

Evaluatio encourages:
- reproducible evaluation pipelines
- transparent statistical methodology
- community review and iteration on best practices

By aligning on common tools and principles, we can make evaluation results more comparable, interpretable, and trustworthy.
