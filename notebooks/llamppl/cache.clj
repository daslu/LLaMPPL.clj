(ns llamppl.cache
  (:require [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [tech.v3.datatype.argops :as argops]
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

(delay
  (->> states-storage
       (map count)
       frequencies))

(md "As an example, let us try the following:
* Use our model context to compute the next word for a piece text.
* Store the model state.
* Use are model with another text.
* Retrieve the states we strored.
* Check the next word again - as it the same as the one in the beginning?")

(delay
  (let [llama-ctx (llms/new-llama-ctx)
        get-next-word (fn [llama-ctx]
                        (-> llama-ctx
                            llama/get-logits
                            argops/argmax
                            vector
                            llms/untokenize))
        ;; Compute the word (recall that llama updates are mutating the context).
        word-at-storage (-> llama-ctx
                            (llama/llama-update "How much wood would a")
                            get-next-word)
        ;; Store the model state
        _ (-> llama-ctx
              (raw/llama_copy_state_data (states-storage 0)))
        ;; Update the model with another text, getting other words.
        another-word (-> llama-ctx
                         (llama/llama-update "How are you")
                         get-next-word)
        ;; Retrieve the model state we stored earlier.
        _ (-> llama-ctx
              (raw/llama_set_state_data (states-storage 0)))
        ;; Compute the next word again
        word-after-retrieval (-> llama-ctx
                                 get-next-word)]
    {:word-at-storage word-at-storage
     :another-word another-word
     :word-after-retrieval word-after-retrieval}))

(md "## FIFO cache
We will manage a basic FIFO cache on top of this storage.

Its API tries to be similar to the
[clojure.cache](https://github.com/clojure/core.cache) API, with some differences.

TODO: Document the cache API better.")

(defn new-fifo-cache []
  {:id->idx {}
   :idx->id {}
   :current-idx 0})

(defn lookup-or-miss!-impl [*fifo-cache id mem-cpy-fn]
  (let [{:keys [id->idx]} @*fifo-cache]
    (or (some-> id
                id->idx
                states-storage)
        (-> *fifo-cache
            (swap! (fn [fifo-cache]
                     (let [updated-fifo-cache
                           (as-> fifo-cache fc
                             (update fc :current-idx
                                     (fn [idx] (-> idx inc (rem n-states))))
                             (update fc :id->idx dissoc ((:idx->id fc)
                                                         (:current-idx fc)))
                             (update fc :id->idx assoc id (:current-idx fc))
                             (update fc :idx->id assoc (:current-idx fc) id))]
                       (-> updated-fifo-cache
                           :current-idx
                           states-storage
                           mem-cpy-fn)
                       updated-fifo-cache)))
            :current-idx
            states-storage))))

(def lookup-or-miss!
  (let [*id (atom 0)]
    (fn [{:keys [state-id
                 llama-ctx-fn
                 *cache]}]
      (let [id (or state-id (swap! *id inc))]
        {:state-id id
         :state-data (lookup-or-miss!-impl
                      *cache
                      id
                      (fn [mem]
                        (raw/llama_copy_state_data
                         (llama-ctx-fn)
                         mem)))}))))

(defn has? [*fifo-cache id]
  (-> @*fifo-cache
      :id->idx
      (contains? id)))

(defn lookup [*fifo-cache id]
  (-> @*fifo-cache
      :id->idx
      (get id)
      states-storage))


(md "As an example, let us try using this cache
with a scenario similar to the one we tried earlier.")

(delay
  (let [llama-ctx (llms/new-llama-ctx)
        get-next-word (fn [llama-ctx]
                        (-> llama-ctx
                            llama/get-logits
                            argops/argmax
                            vector
                            llms/untokenize))
        *cache (atom (new-fifo-cache))
        ;; Use the cache a bit, storing a few states.
        _ (dotimes [i 3]
            (lookup-or-miss!
             {:*cache *cache
              :llama-ctx-fn #(llama/llama-update
                              llama-ctx
                              "How are you")}))
        ;; Use the cache for a text we care about,
        ;; keeping the `state-id` for future reference.
        ;; We also keep the cached `state-data` itself,
        ;; for our testing.
        {:keys [state-id
                state-data]} (lookup-or-miss!
                              {:*cache *cache
                               :llama-ctx-fn #(llama/llama-update
                                               llama-ctx
                                               "How much wood would a")})
        ;; Compute the next word for the text we used.
        word-at-storage (get-next-word llama-ctx)
        ;; Keep updating the model state and using the cache.
        _ (dotimes [i 3]
            (lookup-or-miss!
             {:*cache *cache
              :llama-ctx-fn #(llama/llama-update
                              llama-ctx
                              "How are you")}))
        ;; Compute the next word, which should be another word.
        another-word (get-next-word llama-ctx)
        ;; Retrieve the model state from the cache
        ;; using the `state-id` we remembered.
        retrieved-state-data (lookup *cache state-id)
        ;; Used the retrived state for the model state:
        _ (-> llama-ctx
              (raw/llama_set_state_data
               retrieved-state-data))
        ;; Compate the retrieved state with the state we kept earlier.
        states-comparison (java.util.Arrays/equals
                           ^bytes state-data
                           ^bytes retrieved-state-data)
        ;; Copmute the next word again,
        ;; expecting to get the same the same one we got earlier.
        word-after-retrieval (get-next-word llama-ctx)]
    {:word-at-storage word-at-storage
     :another-word another-word
     :states-comparison states-comparison
     :word-after-retrieval word-after-retrieval}))
