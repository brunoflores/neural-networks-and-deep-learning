# Code samples for "Neural Networks and Deep Learning"

This repository contains code samples for the book ["Neural Networks
and Deep Learning"](http://neuralnetworksanddeeplearning.com) as well as my solutions for some of the problems in it.

## Chapter 2

### Fully matrix-based approach

My solution for the problem in Chapter 2: "Fully matrix-based approach to back-propagation over a mini-batch"
is in [`src/network_matrix_based.py`](src/network_matrix_based.py).

Original implementation by the author performed as `cProfile` shows below:

```text
54861403 function calls (54857353 primitive calls) in 485.136 seconds
```

```text
ncalls   tottime  percall  cumtime   percall  filename:lineno(function)
150000   21.762   0.000    452.468   0.003    network.py:74(update_mini_batch)
```

### Solution

Taking advantage of linear algebra, my proposed approach performed quite a bit faster
than looping over the mini-batch:

```text
16159577 function calls (16155527 primitive calls) in 133.310 seconds
```

```text
ncalls   tottime  percall  cumtime   percall  filename:lineno(function)
150000   2.278    0.000    104.758   0.001    network_matrix_based.py:94(update_mini_batch)
```

## License

MIT License

Copyright (c) 2012-2015 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
