# Evaluatio Changelog

Welcome to the changelog of the library. The Captain's log if you will!

## Release v0.4.0
- Exposed the `error_rate_ci` function through the Python API

## Release v0.5.0 (`babel-fish`)

This release is one of the most substantial yet.
While the release is notable for introducing machine translation metrics and documentation,
but there are significant other improvements, especially to `evaluatio-docs` and the documentation
more generally.

> "All explicit knowledge is translated knowledge, and all translation is imperfect."
>
> -- <cite>Patrick Rothfuss, The Wise Man's Fear</cite>

### Changelog
- Added a bootstrap test for [BLEU](/metrics/bleu) and [ChrF](/metrics/chrf).
- Added a confidence interval function for [BLEU](/metrics/bleu) and [ChrF](/metrics/chrf).
- Added a general purpose permutation test.
- Paired bootstrap test was moved to `evaluatio.inference.hypothesis`.
- Significant improvements to `evaluatio-docs` including:
    - More extensive task guides.
    - Automatically generated API pages.
    - Pages for metrics not implemented by the library like [BLEURT](/metrics/bleurt).
- Added bootstrap_confidence_interval.
- All Python modules now have extensive Numpy style docstrings.

## Release v0.4.0
- Added Bonferroni and Holm-Bonferroni corrections.
- Added article about multiple testing.
- Split the WER article into a separate ASR task article.

## Release v0.3.0
- Added multiprocessing for certain functions using rayon.
- Fixed type annotations to be less restrictive.
- Minor improvements to documentation.
- Minor improvements to CI.

## Release v0.2.0
- `evaluatio-docs` now exists and is hosted on GitHub pages!
- Typing for iterables is now more forgiving (changed List to Iterable).
- Bootstrap based confidence intervals have been added:
    - One generic mean based one.
    - One specifically for WER due to the way corpus-level means are handled.
- Added an independent and paired version of Cohen's d.

## Release v0.1.0
Bonjour! This was the first version of the library and was basically a re-organised version
of the original library [universal-edit-distance](https://pypi.org/project/universal-edit-distance/).
