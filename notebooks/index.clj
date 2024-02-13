^:kindly/hide-code
(ns index)

;; # Preface
;;
;; This repo explores the [LLaMPPL](https://github.com/probcomp/LLaMPPL) underlying algorithms from Clojure using [llama.clj](https://github.com/phronmophobic/llama.clj).
;;
;; At the moment, it demonstrates implementing the Sequential Monte Carlo algoritm on a specific case, the Hard Constraints case generating texts with only short words with a certain choice of M (the Markove kernel) and G (the potential function), not the most efficient one.
;;
;; The main effort so far has been in tackling some of the caching challenges.
