(ns llms
  (:require [tech.v3.datatype :as dtype]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [tech.v3.datatype.argops :as argops]
            [clojure.math :as math]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [scicloj.noj.v1.vis.hanami :as hanami]
            [aerial.hanami.templates :as ht]
            [tablecloth.api :as tc]
            [clojure.string :as str]
            [clojure.walk :as walk]
            [tech.v3.datatype.functional :as fun]))

(def md
  (comp kindly/hide-code kind/md))

(md "
# LLMs: Using llama.clj
")
