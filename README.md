# LLaMPPL.clj

This repo explores the [LLaMPPL](https://github.com/probcomp/LLaMPPL) underlying algorithms from Clojure using [llama.clj](https://github.com/phronmophobic/llama.clj).

Probably, it will be moved under another organisation in case the project seems promising.

## Documentation

[rendered notes](https://daslu.github.io/LLaMPPL.clj)

## Status

WIP

At the moment, we demonstrate implementing the Sequential Monte Carlo algoritm on a specific case, the Hard Constraints case generating texts with only short words with a certain choice of $M$ (the Markov kernel) and $G$ (the potential function), not the most efficient one.

The main effort so far has been in tackling some of the caching challenges.

Everything is considered experimental. Changes are expected.

## Known issues

Some functions crash unexpectedly, sometimes.


## License

The MIT License (MIT)

Copyright © 2023 Daniel Slutsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



