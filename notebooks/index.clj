^:kindly/hide-code
(ns index
  (:require [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]))

^:kindly/hide-code
(def md
  (comp kindly/hide-code kind/md))


(md "
# Preface

This repo explores the [LLaMPPL](https://github.com/probcomp/LLaMPPL) underlying algorithms from Clojure using [llama.clj](https://github.com/phronmophobic/llama.clj).

See [Sequential Monte Carlo Steering of Large Language Models
using Probabilistic Programs](https://arxiv.org/abs/2306.03081)
by Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, Vikash K. Mansinghka
(see Figure 1 and Subsection 2.2).

At the moment, we demonstrate implementing the Sequential Monte Carlo algoritm on a specific case, the Hard Constraints case generating texts with only short words with a certain choice of M (the Markove kernel) and G (the potential function), not the most efficient one.

The main effort so far has been in tackling some of the caching challenges.
")
