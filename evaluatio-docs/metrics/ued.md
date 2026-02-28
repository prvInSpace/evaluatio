# Edit Distance (ED/UED)

Edit distance a metric that measures the number of substitutions, additions, and removals are required to change one array into another. It is also often called the Levenshtein distance due to the person who invented it.

Note that I said array, not a specific type, and that was intentional. The original algorithm described by Levenshtein was correcting binary information [@levenshteinBinaryCodesCapable1966], but nowadays it is often used in NLP to measure character or world level errors (see [](./cer.md) and [](./wer.md)). Despite this, the algorithm only requires elements to be comparable (i.e. they implement `__eq__` in Python). This makes the algorithm very versatile, *universal* if you will.
