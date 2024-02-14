(ns llamppl.trie
  (:require [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [tech.v3.datatype.argops :as argops]
            [clojure.walk :as walk]
            [llamppl.utils :as utils]
            [llamppl.llms :as llms]
            [llamppl.cache :as cache]))

^:kind/hide-code
(def md
  (comp kindly/hide-code kind/md))

(md "# A token-trie cache
Following the LLaMPPL [paper](https://arxiv.org/abs/2306.03081) (Section 3, Subsection \"Shared Transformer cache\"), here we implement a token-trie cache of model states.

As we've seen, we may update the LLM model's state with a sequence of tokens, and then ask the model for the logits of the next token. Typically, we may have already seen a prefix of the given sequence, and may retrieve the model state of that time. The token-trie cache holds such prefix sequences in a tree structure.

## Evaluation
We define the cached evaluation as a recursive transformation of a context map. Note that some parts of the map, spefifically the `llama-ctx` model context and the `*cache` atom, are mutable.

")

(defn cached-eval [{:as context
                    :keys [llama-ctx
                           llama-ctx-state-id
                           trie
                           *cache
                           tokens
                           path
                           sub-trie
                           remaining-tokens]
                    :or {sub-trie trie
                         path []
                         remaining-tokens tokens}}]
  (let [path->text (fn [path]
                     (->> path
                          (filter number?)
                          llms/untokenize))]
    (if (empty? remaining-tokens)
      ;; done - return this context
      (do
        #_(prn [:done (path->text path)])
        context)
      ;; else
      (let [token (first remaining-tokens)
            ;; Look into the next sub trie,
            ;; following this token:
            next-step [:children token]
            next-path (concat path next-step)
            next-sub-trie (get-in sub-trie next-step)]
        (if (some->> next-sub-trie
                     :llama-state-id
                     (cache/has? *cache))
          ;; We have already created next-sub-trie in the past,
          ;; and we have its llama state still in the cache,
          ;; so let us step into it.
          (do
            #_(prn [:recur-known (path->text next-path)])
            (recur (-> context
                       (assoc
                        :sub-trie next-sub-trie
                        :path next-path
                        :remaining-tokens (rest remaining-tokens)
                        :logits (:logits next-sub-trie)))))
          ;; Else, we need to create the next sub trie.
          (let [{:keys [state-id state-data]}
                (cache/lookup-or-miss!
                 {:*cache *cache
                  :llama-ctx-fn (fn []
                                  ;; Make sure the llama-ctx has the right state
                                  ;; to continue.
                                  (cond
                                    ;; When we are in the beginning of the path,
                                    ;; take the base state.
                                    (= path [])
                                    (do
                                      #_(prn [:set-from-base])
                                      (raw/llama_set_state_data llama-ctx
                                                                llms/base-state-data))
                                    ;; When the last evaluation does not fit
                                    ;; out place in the trie,
                                    ;; bring the reletant state from cache.
                                    (-> sub-trie
                                        :llama-state-id
                                        (not= llama-ctx-state-id))
                                    (do
                                      #_(prn [:set-from-cache])
                                      (->> sub-trie
                                           :llama-state-id
                                           (cache/lookup *cache)
                                           (raw/llama_set_state_data llama-ctx)))
                                    ;; Otherwise, our current state is what we need.
                                    :else
                                    (do #_(prn [:continue])
                                        nil))
                                  ;; Evaluate the current token:
                                  (prn [:eval
                                        (path->text path)
                                        '-->
                                        (llms/untokenize [token])])
                                  (time
                                   (llama/llama-update llama-ctx
                                                       token
                                                       ;; n-past
                                                       (->> path
                                                            (filter number?)
                                                            count)
                                                       ;; num-threads
                                                       8))
                                  #_(prn [:extract-state])
                                  llama-ctx)})
                ;; Create the next sub trie:
                new-sub-trie (merge next-sub-trie
                                    {:logits (llama/get-logits llama-ctx)
                                     :llama-state-id state-id})]
            ;; Step into the next sub trie:
            (do #_(prn [:recur-new (path->text next-path)])
                (recur (-> context
                           (update :trie assoc-in next-path new-sub-trie)
                           (assoc :llama-ctx-state-id state-id
                                  :sub-trie new-sub-trie
                                  :path next-path
                                  :remaining-tokens (rest remaining-tokens)
                                  :logits (:logits new-sub-trie)))))))))))

(md "Here is how we create a context map to be passed to the evaluation:")

(defn new-context
  ([]
   (new-context {}))
  ([{:keys [seed]
     :or {seed 12345}}]
   (System/gc)
   (let [llama-ctx (llms/new-llama-ctx)
         samplef (llama/init-mirostat-v2-sampler
                  llama-ctx)]
     (prn [:seed seed])
     (raw/llama_set_rng_seed llama-ctx seed)
     {:llama-ctx llama-ctx
      :samplef samplef
      :trie {}
      :*cache (atom (cache/new-fifo-cache))})))

(md "Given an atom holding a contex map, we may update it with a given cached evaluation.
Note that we do not keep all the fields of the context map.
Some of them (e.g., `:sub-trie`) were used for the recursive evaluation
and need to be discarded (using `select-keys`).")

(defn cached-eval! [*context tokens]
  (let [context (-> @*context
                    (assoc :tokens tokens)
                    cached-eval)]
    (reset! *context
            (select-keys context [:llama-ctx :*cache :samplef :trie :logits]))
    context))

(md "Typically, we not only evaluated a sequence of tokens,
but also ask for the model logits for the next token.")

(defn logits! [*context tokens]
  (-> *context
      (cached-eval! tokens)
      :logits))

(md "For example:")

(delay
  (let [*context (atom (new-context))]
    (->> ["How are"
          "How much wood would a"
          "How much wood could a"]
         (mapv (fn [text]
                 [text '--> (->> text
                                 llms/tokenize
                                 (logits! *context)
                                 argops/argmax
                                 llms/token->str)])))))

(md "## Visualising the trie")

(defn visualize-trie [context]
  (let [{:keys [*cache trie]} context
        *node-id (atom 0)
        *nodes (atom [{:data {:id "0" :word "(root)"}}])
        *edges (atom [])
        trie (-> trie
                 (assoc :node-id (str @*node-id))
                 (->> (walk/prewalk
                       (fn [v]
                         (if (:children v)
                           (-> v
                               (update
                                :children
                                (fn [children]
                                  (->> children
                                       (map
                                        (fn [[token child]]
                                          (let [node-id (str (swap! *node-id inc))]
                                            (swap!
                                             *nodes
                                             conj
                                             {:data {:id node-id
                                                     :token token
                                                     :word (llms/untokenize [token])
                                                     :background (if (->> child
                                                                          :llama-state-id
                                                                          (cache/has? *cache))
                                                                   "lightgreen"
                                                                   "lightgrey")}})
                                            [token (-> child
                                                       (assoc :node-id node-id))])))
                                       (into {})))))
                           v)))
                      (walk/prewalk (fn [v]
                                      (if (:logits v)
                                        (dissoc v :logits)
                                        v)))
                      (walk/prewalk
                       (fn [v]
                         (if-let [{:keys [node-id]} v]
                           (do
                             (->> v
                                  :children
                                  vals
                                  (map
                                   (fn [child]
                                     (let [child-node-id (:node-id child)]
                                       {:data {:id (str node-id "-" child-node-id)
                                               :source node-id
                                               :target child-node-id}})))
                                  (swap! *edges concat))
                             v)
                           v)))))]
    (kind/cytoscape
     {;:trie trie
      :elements {:nodes @*nodes
                 :edges @*edges}
      :style [{:selector "node"
               :css {:content "data(word)"
                     :text-valign "center"
                     :text-halign "center"
                     :height 50
                     :width 50
                     :background-color "data(background)"}}
              {:selector "edge"
               :css {:curve-style "bezier"
                     :target-arrow-shape "triangle"}}]
      :layout {:name "cose"}})))

(md "For example, let us see the trie following a few token sequences.")

(delay
  (let [*context (atom (new-context))]
    [(->> ["How are"
           "How much wood would a"
           "How much wood could a"]
          (mapv (fn [text]
                  [text '--> (->> text
                                  llms/tokenize
                                  (logits! *context)
                                  argops/argmax
                                  llms/token->str)])))
     (visualize-trie @*context)]))