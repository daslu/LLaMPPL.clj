(ns llamppl.llm
  (:require [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [tech.v3.datatype.argops :as argops]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [scicloj.noj.v1.vis.hanami :as hanami]
            [tablecloth.api :as tc]
            [aerial.hanami.templates :as ht]
            [clojure.math :as math]))

^:kind/hide-code
(def md
  (comp kindly/hide-code kind/md))

(md "
# LLMs: Using llama.clj
We will use [llama.clj](https://github.com/phronmophobic/llama.clj), a Clojure wrapper of llama.cpp.
The excellent [llama.clj docs](https://github.com/phronmophobic/llama.clj?tab=readme-ov-file#documentation)
may provide additional background for this notebook.")

;; ## Constants

;; Path to the LLM (assuming the `MODELS_PATH` environment variable is set properly):
(def llama7b-path
  (str (System/getenv "MODELS_PATH")
       "/llama-2-7b-chat.ggmlv3.q4_0.bin"))

;; One megabyte:
(def MB (math/pow 2 20))

;; ## Models

;; Creating a new model context:
(defn new-llama-ctx []
  (llama/create-context
   llama7b-path
   {:use-mlock true}))

;; Let us keep one copy of an unchanged model
;; (to extract basic information):
(def base-llama-ctx
  (new-llama-ctx))

;; ## Tokens

;; A function to turn a String of text to a list of tokens:
(defn tokenize [text]
  (llutil/tokenize base-llama-ctx text))

;; Example:
(delay
  (-> "The Fed says"
      tokenize))

;; A function to turn a list of tokens to a String of text:
(defn untokenize [tokens]
  (llutil/untokenize base-llama-ctx tokens))

;; Example:
(delay
  (-> "The Fed says"
      tokenize
      untokenize))

;; A map from tokens to the corresponding strings:
(def token->str
  (into (sorted-map)
        (comp (map
               (fn [token]
                 [token (raw/llama_token_to_str base-llama-ctx token)]))
              (take-while (fn [[token untoken]]
                            untoken)))
        (range 0 Integer/MAX_VALUE)))

;; Example:
(delay
  (->> "The Fed says"
       tokenize
       (map token->str)))

;; The EOS (end-of-sentence) token:
(def llama-eos (llama/eos base-llama-ctx))

;; Checking whether a sequence of tokens has ended.
(defn finished? [tokens]
  (->> tokens
       (some (partial = llama-eos))
       some?))

;; ## Probabilities

;; Example: Getting next-token logits for a given piece of text.
(delay
  (-> (new-llama-ctx)
      ;; Note thwe are **mutating** the context
      (llama/llama-update "How much wood would a")
      llama/get-logits
      (->> (take 5))
      vec))

;; Let us look at the distribution of logits.

(delay
  (-> (new-llama-ctx)
      (llama/llama-update "How much wood would a")
      llama/get-logits
      (->> (hash-map :logit))
      tc/dataset
      (hanami/histogram :logit {:nbins 100})
      (assoc :height 200)))

;; Example: Picking the next token of the highest probability.

(delay
  (let [llama-ctx (new-llama-ctx)]
    (-> llama-ctx
        (llama/llama-update "How much wood would a")
        llama/get-logits
        argops/argmax
        token->str)))

;; ## Keeping copies of context state

(def state-size
  (-> base-llama-ctx
      (raw/llama_get_state_size)))

;; How big is this state?
(delay
  (-> state-size
      (/ MB)
      (->> (format "%.02f MB"))))

;; Let us keep a copy of the state of our base context:
(def base-state-data
  (let [mem (byte-array state-size)]
    (raw/llama_copy_state_data base-llama-ctx mem)
    mem))
