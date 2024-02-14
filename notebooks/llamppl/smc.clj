(ns llamppl.smc
  (:require [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.functional :as fun]
            [tablecloth.api :as tc]
            [clojure.walk :as walk]
            [clojure.string :as str]
            [llamppl.utils :as utils]
            [llamppl.llms :as llms]
            [llamppl.cache :as cache]
            [llamppl.trie :as trie]))

^:kind/hide-code
(def md
  (comp kindly/hide-code kind/md))

(md "# SMC Sampling of a probabilistic model

Now, let us define a probabilistic model
and apply SMC sampling to it,
following the paper.



## A probabilistic model

The LLaMPPL paper defines a probabilistic model using
an initial state $s_$,
a Markove kernel $M$,
and a potential function $G$.

Here we are implmenting a specific model
of the 'hard constraints' type:
generating texts that use only short words.

Note that the paper offers more than one option
for the Markov kernel $M$ and the potential function $G$.
For now, we are not using the most efficient choice of them.

### The Markov kernel
We define M as a sampling step
(which is the way we use it algorithmically).
")

(defn M-step [*context
              previous-tokens]
  (if (llms/finished? previous-tokens)
    previous-tokens
    (->> previous-tokens
         (trie/logits! *context)
         (trie/sample-once! *context)
         (conj previous-tokens))))

(md "For example:")

(delay
  (let [*context (atom (trie/new-context {:seed 1}))]
    [(->> #(->> "How much wood"
                llms/tokenize
                (iterate (partial M-step *context))
                (take 5)
                last
                llms/untokenize)
          (repeatedly 5)
          vec)
     (trie/visualize-trie @*context)]))


(delay
  (let [*context (atom (trie/new-context {:seed 1}))]
    [(->> #(->> "How much wood"
                llms/tokenize
                (iterate (partial M-step *context))
                (take 40)
                last
                llms/untokenize)
          (repeatedly 2)
          vec)
     (trie/visualize-trie @*context)]))

(delay
  (let [*context (atom (trie/new-context {:seed 1}))]
    [(->> (fn []
            [(->> "The Fed says"
                  llms/tokenize
                  (iterate (partial M-step *context))
                  (take 20)
                  (map (juxt llms/finished?
                             llms/untokenize))
                  (map-indexed vector)
                  vec)
             (trie/visualize-trie @*context)])
          (repeatedly 10)
          vec)]))

;; ### The potential function

(defn G [threshold current-tokens]
  (if (-> current-tokens
          llms/untokenize
          (str/split  #" ")
          (->> (every? #(-> % count (<= threshold)))))
    1 0))


(defn normalize [ws]
  (fun// ws
         (fun/sum ws)))

(defn find-c [weights N]
  (prn [:weights weights
        :N N])
  (let [sorted-weights (vec (sort weights))]
    (loop [B-val 0.0
           A-val (count weights)
           i 0]
      (let [chi (sorted-weights i)
            new-A-val (dec A-val)
            new-B-val (+ B-val chi)]
        (if (= i N)
          N
          (if (-> new-B-val
                  (/ chi)
                  (+ new-A-val)
                  (- N)
                  (<= 1e-12))
            (/ (- N new-A-val)
               new-B-val)
            (recur new-B-val
                   new-A-val
                   (inc i))))))))

(delay
  (find-c [0.1 0.2] 10))

(delay
  (let [*context (atom (trie/new-context {:seed 1}))]
    (->> "Please complete the sentence in ten words. Clojure is a"
         llms/tokenize
         (M-step *context)
         (M-step *context)
         (M-step *context)
         ((juxt llms/untokenize
                (partial G 9)
                (partial G 12))))))


(defn new-smc-state [] {:stop false
                        :particles []})

(defn run-smc!
  [*smc-state
   {:keys [cache-threshold
           seed
           max-token-length
           N
           K
           base-text
           initial-N
           max-text-length]}]
  (let [*context (atom (trie/new-context {:seed 1}))
        s0 (llms/tokenize base-text)]
    (swap! *smc-state
           assoc :particles  (tc/dataset {:x (repeat initial-N s0)
                                          :w 1
                                          :time (repeat initial-N (utils/now))
                                          :gen 0}))
    (loop [gen 1]
      (let [particles (:particles @*smc-state)
            finished (->> particles
                          :x
                          (map (fn [x]
                                 (or (llms/finished? x)
                                     (-> x count (>= max-text-length))))))]
        (->> finished
             frequencies
             (vector :finished-freqs)
             prn)
        (if (or (:stop @*smc-state)
                (every? true? finished))
          {:particles particles
           :Z (-> particles :w fun/mean)}
          ;; else
          (let [K (->> finished
                       (map (fn [f]
                              (if f 1 K))))
                N-prime (fun/sum K)
                new-particles (-> particles
                                  (tc/add-columns {:K K
                                                   :finished finished})
                                  (tc/rows :as-maps)
                                  (->> (map (fn [{:keys [x w finished K]
                                                  :as row}]
                                              (if finished
                                                (tc/dataset {:x [x]
                                                             :w [(* w N-prime (/ N))]
                                                             :time [(:time row)]
                                                             :gen [(:gen row)]})
                                                ;; else
                                                (-> (range K)
                                                    (->> (map (fn [k]
                                                                (-> {:x (M-step *context x)
                                                                     :time (utils/now)
                                                                     :gen gen}))))
                                                    tc/dataset
                                                    (tc/map-columns
                                                     :w
                                                     [:x]
                                                     (fn [x]
                                                       (* (/ N-prime
                                                             (* K N))
                                                          w
                                                          (G max-token-length x))))))))
                                       (apply tc/concat))
                                  (tc/add-column :w #(-> % :w normalize))
                                  ((fn [{:keys [x w time gen]
                                         :as new-particles}]
                                     (prn [:new-particles new-particles])
                                     (let [w-sum (fun/sum w)
                                           c* (find-c w N)
                                           indexes (-> new-particles
                                                       tc/row-count
                                                       range)
                                           I-det (->> indexes
                                                      (filter (fn [i]
                                                                (-> i
                                                                    w
                                                                    (* c*)
                                                                    (>= 1)))))
                                           I-stoch (->> indexes
                                                        (filter (fn [i]
                                                                  (-> i
                                                                      w
                                                                      (* c*)
                                                                      (< 1))))
                                                        vec)
                                           alpha (/ (->> I-stoch
                                                         (map w)
                                                         fun/sum)
                                                    (- N (count I-det)))
                                           I-strat (loop [candidates I-stoch
                                                          U (* alpha (rand))
                                                          I-strat []]
                                                     (if (empty? candidates)
                                                       I-strat
                                                       (let [i (first candidates)
                                                             U (- U (w i))]
                                                         (if (neg? U)
                                                           (recur (rest candidates)
                                                                  (+ U alpha)
                                                                  (conj I-strat i))
                                                           (recur (rest candidates)
                                                                  U
                                                                  I-strat)))))]
                                       (prn [:c* c*
                                             :I-det I-det
                                             :I-stoch I-stoch
                                             :I-strat I-strat])
                                       (tc/dataset
                                        (concat (->> I-det
                                                     (map (fn [i]
                                                            {:x (x i)
                                                             :w (* (w i)
                                                                   (/ N N-prime))
                                                             :time (time i)
                                                             :gen (gen i)})))
                                                (->> I-strat
                                                     (map (fn [i]
                                                            {:x (x i)
                                                             :w (* (/ N N-prime c*)
                                                                   w-sum)
                                                             :time (time i)
                                                             :gen (gen i)})))))))))]
            (swap! *smc-state
                   assoc :particles new-particles)
            (recur (inc gen))))))))


#_(delay
    (let [*smc-state (atom (new-smc-state))]
      (run-smc! *smc-state
                {:cache-threshold 30
                 :seed 1
                 :base-text "The Fed says"
                 :max-token-length 5
                 :N 15
                 :K 3
                 :initial-N 5
                 :max-text-length 10})
      (-> @*smc-state
          :particles
          (tc/map-columns :finished
                          [:x]
                          llms/finished?)
          (tc/map-columns :length
                          [:x]
                          count)
          (tc/map-columns :x
                          [:x]
                          llms/untokenize)
          (tech.v3.dataset.print/print-range :all))))
