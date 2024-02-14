^:kindly/hide-code
(ns index
  (:require [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [llamppl.smc :as smc]
            [tablecloth.api :as tc]))

^:kindly/hide-code
(def md
  (comp kindly/hide-code kind/md))


(md "
# Preface

This repo explores the [LLaMPPL](https://github.com/probcomp/LLaMPPL) underlying algorithms from Clojure using [llama.clj](https://github.com/phronmophobic/llama.clj).

See [Sequential Monte Carlo Steering of Large Language Models
using Probabilistic Programs](https://arxiv.org/abs/2306.03081)
by Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, Vikash K. Mansinghka.

At the moment, we demonstrate implementing the Sequential Monte Carlo algorithm on a specific case, the Hard Constraints case (See Figure 1 and Subection 2.2 of the paper). Specifically, we implement the \"only short words\" constraint using a certain choice of $M$ (the Markove kernel) and $G$ (the potential function) (but not the most efficient choice for that case).

The main effort so far has been in tackling some of the caching challenges.

Our current bottom line is being able to generate
a random population of texts (\"particles\" in SMC jargon)
starting with \"The Fed says\"
and using only words of at most 5 letters.")


(require '[llamppl.llm :as llm]
         '[llamppl.smc :as smc]
         '[tablecloth.api :as tc])


(delay
  (let [*smc-state (atom (smc/new-smc-state))]
    (smc/run-smc! *smc-state
                  {:cache-threshold 30
                   :seed 1
                   :base-text "The Fed says"
                   :max-n-letters 5
                   :N 10
                   :K 3
                   :initial-N 5
                   :max-text-length 10})
    (-> @*smc-state
        :particles
        (tc/map-columns :finished [:x] llm/finished?)
        (tc/map-columns :length [:x] count)
        (tc/map-columns :text [:x] llm/untokenize)
        (tc/drop-columns [:x])
        (tc/set-dataset-name "texts")
        (tech.v3.dataset.print/print-range :all))))
