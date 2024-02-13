(ns llamppl.cache
  (:require [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [llamppl.llms :as llms]))

^:kind/hide-code
(def md
  (comp kindly/hide-code kind/md))

(md "# Caching model state

We will rely on the model's internal [KV-caching](https://www.dipkumar.dev/becoming-the-unbeatable/posts/gpt-kvcache/). Also, following the LLaMPPL [paper](https://arxiv.org/abs/2306.03081) (Section 3, Subsection \"Shared Transformer cache\"), we will implement a trie cache of model states.

Since the model states are expensive, we cannot store an unlimited number of model states. many of them, so we will need a basic storage space for a few of them.

To avoid garbage collection, we will allocate our memory slots space once, and manage them as a simple FIFO storage.

## A storage space

;; Let us create a space to store a few model states. For now, the number of slots we use is hardcoded, fitting our JVM heap space.")

(def n-states 70)

(defonce states-storage
  (vec (repeatedly
        n-states
        #(byte-array llms/state-size))))
