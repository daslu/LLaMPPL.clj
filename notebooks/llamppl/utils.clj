(ns llamppl.utils
  (:require [tech.v3.datatype.functional :as fun]))

;; # A few utility functions

;; ## Time

;; Getting the time now (useful for reporting):
(defn now []
  (java.util.Date.))

;; For example:
(delay
  (now))

;; ## Math

;; Normalizing a vector or array of numbers
;; so that their sum would be `1`:
(defn normalize [ws]
  (fun// ws
         (fun/sum ws)))

;; For example:

(delay
  (normalize [1 3 3 1]))
