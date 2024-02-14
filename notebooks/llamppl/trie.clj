(ns llamppl.trie
  (:require [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [tech.v3.datatype.argops :as argops]
            [clojure.walk :as walk]
            [llamppl.utils :as utils]
            [llamppl.llm :as llm]
            [llamppl.cache :as cache]))

^:kind/hide-code
(def md
  (comp kindly/hide-code kind/md))

(md "# A token-trie cache
Following the LLaMPPL [paper](https://arxiv.org/abs/2306.03081) (Section 3, Subsection \"Shared Transformer cache\"), here we implement a token-trie cache of model states.

As we've seen, we may update the LLM model's state with a sequence of tokens, and then ask the model for the logits of the next token. Typically, we may have already seen a prefix of the given sequence, and may reuse the model state of that time. The token-trie cache holds such prefix sequences, and their corresponding model states, in a tree structure.

## Evaluation
We define cached evaluation as a recursive transformation of a context map. Note that it is not a pure functional transformation, as some parts of the map, spefifically the `llama-ctx` model context and the `*cache` atom, are mutable.

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
                          llm/untokenize))]
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
                                                                llm/base-state-data))
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
                                        (llm/token->str token)])
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
   (let [llama-ctx (llm/new-llama-ctx)
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

(md "Typically, we not only update the model with a sequence of tokens,
but are also interested in the model logits for the next token.")

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
                                 llm/tokenize
                                 (logits! *context)
                                 argops/argmax
                                 llm/token->str)])))))

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
                                                     :word (llm/token->str token)
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
                                  llm/tokenize
                                  (logits! *context)
                                  argops/argmax
                                  llm/token->str)])))
     (visualize-trie @*context)]))

(md "Here is a bigger example.
Note that some nodes are no longer in the cache
and are coloured differently.
([Source](https://en.wikipedia.org/wiki/Groundhog))")

(delay
  (let [*context (atom (new-context))]
    [(->> ["The groundhog (Marmota monax), also known as the woodchuck, is a rodent of the family Sciuridae, belonging to the group of large ground squirrels known as marmots. The groundhog is a lowland creature of North America; it is found through much of the Eastern United States, across Canada and into Alaska. It was first scientifically described by Carl Linnaeus in 1758."
           "The groundhog is also referred to as a chuck, wood-shock, groundpig, whistlepig, whistler, thickwood badger, Canada marmot, monax, moonack, weenusk, red monk, land beaver, and, among French Canadians in eastern Canada, siffleux. The name \"thickwood badger\" was given in the Northwest to distinguish the animal from the prairie badger. Monax (Móonack) is an Algonquian name of the woodchuck, which means \"digger\" (cf. Lenape monachgeu). Young groundhogs may be called chucklings. "
           "The groundhog did visit me yesterday."
           "The groundhog is also referred to as Margaret. At least that is how they call her in our neighbourhood."]
          (mapv (fn [text]
                  [text '--> (->> text
                                  llm/tokenize
                                  (logits! *context)
                                  argops/argmax
                                  llm/token->str)])))
     (visualize-trie @*context)]))


(md "## Sampling random tokens

Our context holds a sample function `samplef`,
which allows us to sample tokens according to logits.

For example:")


(delay
  (let [*context (atom (new-context {:seed 1}))
        {:keys [samplef]} @*context]
    (->> ["How much wood would a"
          "How much wood would a woodchuck"
          "How much wood would a woodchuck chuck"]
         (mapv
          (fn [text]
            (let [logits (->> text
                              llm/tokenize
                              (logits! *context))]
              [text
               (->> (repeatedly
                     1000
                     #(llm/token->str (samplef logits)))
                    frequencies)]))))))

(md "For convenience, let us use this function to sample one token.
Note that we change the seed for diversity.

TODO: Handle seeds more carefully for reproducibility..")

(defn sample-once! [*context logits]
  (-> @*context
      :llama-ctx
      (raw/llama_set_rng_seed (rand-int 9999)))
  ((:samplef @*context)
   logits))

(md "For example:")

(delay
  (let [*context (atom (new-context))]
    (->> "How much wood would a"
         llm/tokenize
         (logits! *context)
         (sample-once! *context)
         llm/token->str)))
